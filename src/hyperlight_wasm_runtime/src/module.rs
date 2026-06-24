/*
Copyright 2024 The Hyperlight Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::ops::{Deref, DerefMut};

use hyperlight_common::flatbuffer_wrappers::function_call::FunctionCall;
use hyperlight_common::flatbuffer_wrappers::function_types::{
    ParameterType, ParameterValue, ReturnType,
};
use hyperlight_common::flatbuffer_wrappers::guest_error::ErrorCode;
use hyperlight_common::flatbuffer_wrappers::util::get_flatbuffer_result;
use hyperlight_guest::error::{HyperlightGuestError, Result};
use hyperlight_guest_bin::guest_function::definition::GuestFunctionDefinition;
use hyperlight_guest_bin::guest_function::register::register_function;
use hyperlight_guest_bin::host_comm::print_output_with_host_print;
use spin::Mutex;
use tracing::instrument;
use wasmtime::{Config, Engine, Linker, Module, Store, Val};
#[cfg(feature = "snapshot-linear-mem")]
use wasmtime::{Extern, Global, Mutability};

use crate::{hostfuncs, map_wasmtime_error, marshal, platform, wasip1};

// Set by transition to WasmSandbox (by init_wasm_runtime)
static CUR_ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
static CUR_LINKER: Mutex<Option<Linker<()>>> = Mutex::new(None);
// Set by transition to LoadedWasmSandbox (by load_wasm_module/load_wasm_module_phys)
static CUR_MODULE: Mutex<Option<Module>> = Mutex::new(None);
static CUR_STORE: Mutex<Option<Store<()>>> = Mutex::new(None);
static CUR_INSTANCE: Mutex<Option<wasmtime::Instance>> = Mutex::new(None);

#[no_mangle]
#[instrument(skip_all, level = "Info")]
pub fn guest_dispatch_function(function_call: FunctionCall) -> Result<Vec<u8>> {
    let mut store = CUR_STORE.lock();
    let store = store.deref_mut().as_mut().ok_or(HyperlightGuestError::new(
        ErrorCode::GuestError,
        "No wasm store available".to_string(),
    ))?;
    let instance = CUR_INSTANCE.lock();
    let instance = instance.deref().as_ref().ok_or(HyperlightGuestError::new(
        ErrorCode::GuestError,
        "No wasm instance available".to_string(),
    ))?;

    // Free any return value allocations from the previous VM call
    // This implements the memory model where hyperlight owns return values
    // and frees them on the next VM entry
    marshal::free_return_value_allocations(&mut *store, &|ctx, name| {
        instance.get_export(ctx, name)
    })?;

    let func = instance
        .get_func(&mut *store, &function_call.function_name)
        .ok_or(HyperlightGuestError::new(
            ErrorCode::GuestError,
            "Function not found".to_string(),
        ))?;

    let mut w_params = vec![];
    for f_param in (function_call.parameters)
        .as_ref()
        .unwrap_or(&vec![])
        .iter()
    {
        w_params.push(marshal::hl_param_to_val(
            &mut *store,
            |ctx, name| instance.get_export(ctx, name),
            f_param,
        )?);
    }
    let is_void = ReturnType::Void == function_call.expected_return_type;
    let n_results = if is_void { 0 } else { 1 };
    let mut results = vec![Val::I32(0); n_results];
    func.call(&mut *store, &w_params, &mut results)
        .map_err(map_wasmtime_error)?;
    marshal::val_to_hl_result(
        &mut *store,
        |ctx, name| instance.get_export(ctx, name),
        function_call.expected_return_type,
        &results,
    )
}

#[instrument(skip_all, level = "Info")]
fn init_wasm_runtime(function_call: FunctionCall) -> Result<Vec<u8>> {
    let mut config = Config::new();
    // Enable x86_float_abi_ok only for the latest Wasmtime native x86 target.
    // Safety:
    // We are using hyperlight cargo to build the guest which
    // sets the Rust target to be compiled with the hard-float ABI manually via
    // `-Zbuild-std` and a custom target JSON configuration
    // See https://github.com/bytecodealliance/wasmtime/pull/11553
    #[cfg(all(not(feature = "wasmtime_lts"), not(pulley)))]
    unsafe {
        config.x86_float_abi_ok(true)
    };

    config.with_custom_code_memory(Some(alloc::sync::Arc::new(platform::WasmtimeCodeMemory {})));
    #[cfg(gdb)]
    config.debug_info(true);
    #[cfg(pulley)]
    config.target("pulley64").map_err(|_| {
        HyperlightGuestError::new(
            ErrorCode::GuestError,
            "Failed to set wasmtime target: pulley64".to_string(),
        )
    })?;
    let engine = Engine::new(&config).map_err(map_wasmtime_error)?;
    let mut linker = Linker::new(&engine);
    wasip1::register_handlers(&mut linker)?;

    // Parse host function details pushed by the host as a parameter
    let params = function_call.parameters.as_ref().ok_or_else(|| {
        HyperlightGuestError::new(
            ErrorCode::GuestFunctionParameterTypeMismatch,
            "InitWasmRuntime: missing parameters".to_string(),
        )
    })?;

    let bytes = match params.first() {
        Some(ParameterValue::VecBytes(ref b)) => b,
        Some(_) => {
            return Err(HyperlightGuestError::new(
                ErrorCode::GuestFunctionParameterTypeMismatch,
                "InitWasmRuntime: first parameter must be VecBytes".to_string(),
            ))
        }
        None => {
            return Err(HyperlightGuestError::new(
                ErrorCode::GuestFunctionParameterTypeMismatch,
                "InitWasmRuntime: expected 1 parameter, got 0".to_string(),
            ))
        }
    };

    let hfd: hostfuncs::HostFunctionDetails = bytes.as_slice().try_into().map_err(|e| {
        HyperlightGuestError::new(
            ErrorCode::GuestError,
            alloc::format!("Failed to parse host function details: {:?}", e),
        )
    })?;
    let hostfuncs = hfd.host_functions.unwrap_or_default();

    for hostfunc in hostfuncs.iter() {
        let captured = hostfunc.clone();
        linker
            .func_new(
                "env",
                &hostfunc.function_name,
                hostfuncs::hostfunc_type(hostfunc, &engine)?,
                move |c, ps, rs| {
                    hostfuncs::call(&captured, c, ps, rs)
                        .map_err(|e| wasmtime::Error::msg(format!("{:?}", e)))
                },
            )
            .map_err(map_wasmtime_error)?;
    }

    *CUR_ENGINE.lock() = Some(engine);
    *CUR_LINKER.lock() = Some(linker);
    Ok(get_flatbuffer_result::<i32>(0))
}

#[instrument(skip_all, level = "Info")]
fn load_wasm_module(function_call: FunctionCall) -> Result<Vec<u8>> {
    if let (
        ParameterValue::VecBytes(ref wasm_bytes),
        ParameterValue::Int(ref _len),
        Some(ref engine),
    ) = (
        &function_call.parameters.as_ref().unwrap()[0],
        &function_call.parameters.as_ref().unwrap()[1],
        &*CUR_ENGINE.lock(),
    ) {
        let linker = CUR_LINKER.lock();
        let linker = linker.deref().as_ref().ok_or(HyperlightGuestError::new(
            ErrorCode::GuestError,
            "impossible: wasm runtime has no valid linker".to_string(),
        ))?;

        let module =
            unsafe { Module::deserialize(engine, wasm_bytes).map_err(map_wasmtime_error)? };
        let mut store = Store::new(engine, ());
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(map_wasmtime_error)?;

        *CUR_MODULE.lock() = Some(module);
        *CUR_STORE.lock() = Some(store);
        *CUR_INSTANCE.lock() = Some(instance);
        Ok(get_flatbuffer_result::<i32>(0))
    } else {
        Err(HyperlightGuestError::new(
            ErrorCode::GuestFunctionParameterTypeMismatch,
            "Invalid parameters passed to LoadWasmModule".to_string(),
        ))
    }
}

#[instrument(skip_all, level = "Info")]
fn load_wasm_module_phys(function_call: FunctionCall) -> Result<Vec<u8>> {
    if let (ParameterValue::ULong(ref phys), ParameterValue::ULong(ref len), Some(ref engine)) = (
        &function_call.parameters.as_ref().unwrap()[0],
        &function_call.parameters.as_ref().unwrap()[1],
        &*CUR_ENGINE.lock(),
    ) {
        let linker = CUR_LINKER.lock();
        let linker = linker.deref().as_ref().ok_or(HyperlightGuestError::new(
            ErrorCode::GuestError,
            "impossible: wasm runtime has no valid linker".to_string(),
        ))?;

        let module = unsafe {
            Module::deserialize_raw(engine, platform::map_buffer(*phys, *len))
                .map_err(map_wasmtime_error)?
        };
        let mut store = Store::new(engine, ());
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(map_wasmtime_error)?;

        *CUR_MODULE.lock() = Some(module);
        *CUR_STORE.lock() = Some(store);
        *CUR_INSTANCE.lock() = Some(instance);
        Ok(get_flatbuffer_result::<()>(()))
    } else {
        Err(HyperlightGuestError::new(
            ErrorCode::GuestFunctionParameterTypeMismatch,
            "Invalid parameters passed to LoadWasmModulePhys".to_string(),
        ))
    }
}

#[cfg(feature = "snapshot-linear-mem")]
const WASM_PAGE_SIZE: usize = 65536;

#[cfg(feature = "snapshot-linear-mem")]
fn snapshot_state_err(msg: &str) -> HyperlightGuestError {
    HyperlightGuestError::new(ErrorCode::GuestError, msg.to_string())
}

/// Returns the size in bytes of the loaded guest's default linear memory.
///
/// The host loops `SnapshotMemSize` + `SnapshotReadMem` to assemble the whole
/// image in chunks that fit the (default 16 KiB) ABI buffer.
#[cfg(feature = "snapshot-linear-mem")]
#[instrument(skip_all, level = "Info")]
fn snapshot_mem_size(_function_call: FunctionCall) -> Result<Vec<u8>> {
    let mut store = CUR_STORE.lock();
    let store = store
        .deref_mut()
        .as_mut()
        .ok_or_else(|| snapshot_state_err("no wasm store"))?;
    let instance = CUR_INSTANCE.lock();
    let instance = instance
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no wasm instance"))?;
    let mem = instance
        .get_export(&mut *store, "memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| snapshot_state_err("guest does not export 'memory'"))?;
    let size = mem.data_size(&*store) as u64;
    Ok(get_flatbuffer_result::<u64>(size))
}

/// Reads `len` bytes of linear memory starting at `offset` (bounds-checked).
#[cfg(feature = "snapshot-linear-mem")]
#[instrument(skip_all, level = "Info")]
fn snapshot_read_mem(function_call: FunctionCall) -> Result<Vec<u8>> {
    let params = function_call.parameters.unwrap_or_default();
    let (offset, len) = match (params.first(), params.get(1)) {
        (Some(ParameterValue::ULong(o)), Some(ParameterValue::ULong(l))) => {
            (*o as usize, *l as usize)
        }
        _ => {
            return Err(HyperlightGuestError::new(
                ErrorCode::GuestFunctionParameterTypeMismatch,
                "SnapshotReadMem expects (ULong offset, ULong len)".to_string(),
            ))
        }
    };
    let mut store = CUR_STORE.lock();
    let store = store
        .deref_mut()
        .as_mut()
        .ok_or_else(|| snapshot_state_err("no wasm store"))?;
    let instance = CUR_INSTANCE.lock();
    let instance = instance
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no wasm instance"))?;
    let mem = instance
        .get_export(&mut *store, "memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| snapshot_state_err("guest does not export 'memory'"))?;
    let total = mem.data_size(&*store);
    let end = offset
        .checked_add(len)
        .ok_or_else(|| snapshot_state_err("SnapshotReadMem range overflow"))?;
    if end > total {
        return Err(snapshot_state_err(&format!(
            "SnapshotReadMem out of range: offset={offset} len={len} total={total}"
        )));
    }
    let mut buf = vec![0u8; len];
    mem.read(&*store, offset, &mut buf)
        .map_err(|e| snapshot_state_err(&format!("SnapshotReadMem failed: {e}")))?;
    Ok(get_flatbuffer_result::<&[u8]>(&buf))
}

/// Grows linear memory so it is at least `size_bytes` (no-op if already large
/// enough). Memory can only grow; the caller restores into a fresh instance
/// (via `SnapshotReset`) before writing a smaller image back.
#[cfg(feature = "snapshot-linear-mem")]
#[instrument(skip_all, level = "Info")]
fn snapshot_grow_mem_to(function_call: FunctionCall) -> Result<Vec<u8>> {
    let params = function_call.parameters.unwrap_or_default();
    let target = match params.first() {
        Some(ParameterValue::ULong(s)) => *s as usize,
        _ => {
            return Err(HyperlightGuestError::new(
                ErrorCode::GuestFunctionParameterTypeMismatch,
                "SnapshotGrowMemTo expects (ULong size_bytes)".to_string(),
            ))
        }
    };
    let mut store = CUR_STORE.lock();
    let store = store
        .deref_mut()
        .as_mut()
        .ok_or_else(|| snapshot_state_err("no wasm store"))?;
    let instance = CUR_INSTANCE.lock();
    let instance = instance
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no wasm instance"))?;
    let mem = instance
        .get_export(&mut *store, "memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| snapshot_state_err("guest does not export 'memory'"))?;
    let cur = mem.data_size(&*store);
    if target > cur {
        let delta_pages = (target - cur).div_ceil(WASM_PAGE_SIZE);
        mem.grow(&mut *store, delta_pages as u64)
            .map_err(|e| snapshot_state_err(&format!("SnapshotGrowMemTo failed: {e}")))?;
    }
    Ok(get_flatbuffer_result::<()>(()))
}

/// Writes `data` into linear memory at `offset` (bounds-checked; grow first).
#[cfg(feature = "snapshot-linear-mem")]
#[instrument(skip_all, level = "Info")]
fn snapshot_write_mem(function_call: FunctionCall) -> Result<Vec<u8>> {
    let params = function_call.parameters.unwrap_or_default();
    let (offset, data) = match (params.first(), params.get(1)) {
        (Some(ParameterValue::ULong(o)), Some(ParameterValue::VecBytes(b))) => (*o as usize, b),
        _ => {
            return Err(HyperlightGuestError::new(
                ErrorCode::GuestFunctionParameterTypeMismatch,
                "SnapshotWriteMem expects (ULong offset, VecBytes data)".to_string(),
            ))
        }
    };
    let mut store = CUR_STORE.lock();
    let store = store
        .deref_mut()
        .as_mut()
        .ok_or_else(|| snapshot_state_err("no wasm store"))?;
    let instance = CUR_INSTANCE.lock();
    let instance = instance
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no wasm instance"))?;
    let mem = instance
        .get_export(&mut *store, "memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| snapshot_state_err("guest does not export 'memory'"))?;
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| snapshot_state_err("SnapshotWriteMem range overflow"))?;
    if end > mem.data_size(&*store) {
        return Err(snapshot_state_err(
            "SnapshotWriteMem out of range (call SnapshotGrowMemTo first)",
        ));
    }
    mem.write(&mut *store, offset, data)
        .map_err(|e| snapshot_state_err(&format!("SnapshotWriteMem failed: {e}")))?;
    Ok(get_flatbuffer_result::<()>(()))
}

/// Captures ALL exported mutable globals as a self-describing blob.
///
/// Generic across guests (not hardcoded to `__stack_pointer`). Layout:
/// `[u32 count]` then per entry `[u32 name_len][name][u8 tag][u8;8 payload]`,
/// tag 0=i32 1=i64 2=f32(bits) 3=f64(bits), payload little-endian. Fails closed
/// on any mutable global whose type is not a captured numeric scalar (e.g.
/// mutable reference/v128) since those cannot be faithfully serialized here.
#[cfg(feature = "snapshot-linear-mem")]
#[instrument(skip_all, level = "Info")]
fn snapshot_get_globals(_function_call: FunctionCall) -> Result<Vec<u8>> {
    let mut store = CUR_STORE.lock();
    let store = store
        .deref_mut()
        .as_mut()
        .ok_or_else(|| snapshot_state_err("no wasm store"))?;
    let instance = CUR_INSTANCE.lock();
    let instance = instance
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no wasm instance"))?;

    // Collect (name, handle) first so the exports() store borrow ends before we
    // start reading each global's value.
    let globals: Vec<(alloc::string::String, Global)> = instance
        .exports(&mut *store)
        .filter_map(|e| {
            let name = e.name().to_string();
            match e.into_extern() {
                Extern::Global(g) => Some((name, g)),
                _ => None,
            }
        })
        .collect();

    let mut body: Vec<u8> = Vec::new();
    let mut count: u32 = 0;
    for (name, g) in &globals {
        if g.ty(&*store).mutability() != Mutability::Var {
            continue;
        }
        let (tag, payload): (u8, [u8; 8]) = match g.get(&mut *store) {
            Val::I32(v) => {
                let mut p = [0u8; 8];
                p[..4].copy_from_slice(&v.to_le_bytes());
                (0, p)
            }
            Val::I64(v) => (1, v.to_le_bytes()),
            Val::F32(bits) => {
                let mut p = [0u8; 8];
                p[..4].copy_from_slice(&bits.to_le_bytes());
                (2, p)
            }
            Val::F64(bits) => (3, bits.to_le_bytes()),
            other => {
                return Err(snapshot_state_err(&format!(
                    "SnapshotGetGlobals: mutable global '{name}' has unsupported type {other:?}"
                )))
            }
        };
        let name_bytes = name.as_bytes();
        body.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(name_bytes);
        body.push(tag);
        body.extend_from_slice(&payload);
        count += 1;
    }
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(get_flatbuffer_result::<&[u8]>(&out))
}

/// Restores mutable globals from a blob produced by `SnapshotGetGlobals`.
#[cfg(feature = "snapshot-linear-mem")]
#[instrument(skip_all, level = "Info")]
fn snapshot_set_globals(function_call: FunctionCall) -> Result<Vec<u8>> {
    let params = function_call.parameters.unwrap_or_default();
    let blob = match params.first() {
        Some(ParameterValue::VecBytes(b)) => b,
        _ => {
            return Err(HyperlightGuestError::new(
                ErrorCode::GuestFunctionParameterTypeMismatch,
                "SnapshotSetGlobals expects (VecBytes globals)".to_string(),
            ))
        }
    };
    if blob.len() < 4 {
        return Err(snapshot_state_err("SnapshotSetGlobals: truncated blob"));
    }
    let count = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
    let mut pos = 4usize;

    let mut store = CUR_STORE.lock();
    let store = store
        .deref_mut()
        .as_mut()
        .ok_or_else(|| snapshot_state_err("no wasm store"))?;
    let instance = CUR_INSTANCE.lock();
    let instance = instance
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no wasm instance"))?;

    for _ in 0..count {
        if pos + 4 > blob.len() {
            return Err(snapshot_state_err("SnapshotSetGlobals: truncated blob"));
        }
        let name_len =
            u32::from_le_bytes([blob[pos], blob[pos + 1], blob[pos + 2], blob[pos + 3]]) as usize;
        pos += 4;
        if pos + name_len + 1 + 8 > blob.len() {
            return Err(snapshot_state_err("SnapshotSetGlobals: truncated blob"));
        }
        let name = core::str::from_utf8(&blob[pos..pos + name_len])
            .map_err(|e| snapshot_state_err(&format!("SnapshotSetGlobals: bad name utf8: {e}")))?
            .to_string();
        pos += name_len;
        let tag = blob[pos];
        pos += 1;
        let mut payload = [0u8; 8];
        payload.copy_from_slice(&blob[pos..pos + 8]);
        pos += 8;
        let val = match tag {
            0 => Val::I32(i32::from_le_bytes([
                payload[0], payload[1], payload[2], payload[3],
            ])),
            1 => Val::I64(i64::from_le_bytes(payload)),
            2 => Val::F32(u32::from_le_bytes([
                payload[0], payload[1], payload[2], payload[3],
            ])),
            3 => Val::F64(u64::from_le_bytes(payload)),
            t => {
                return Err(snapshot_state_err(&format!(
                    "SnapshotSetGlobals: unknown tag {t}"
                )))
            }
        };
        let g = instance.get_global(&mut *store, &name).ok_or_else(|| {
            snapshot_state_err(&format!("SnapshotSetGlobals: global '{name}' not found"))
        })?;
        if g.ty(&*store).mutability() != Mutability::Var {
            return Err(snapshot_state_err(&format!(
                "SnapshotSetGlobals: global '{name}' is not mutable"
            )));
        }
        g.set(&mut *store, val).map_err(|e| {
            snapshot_state_err(&format!("SnapshotSetGlobals: set '{name}' failed: {e}"))
        })?;
    }
    Ok(get_flatbuffer_result::<()>(()))
}

/// Re-instantiates the loaded module in a FRESH store, resetting linear memory
/// to the module minimum. This is the state-reset primitive: restore =
/// `SnapshotReset` then grow + write memory + set globals (mirrors quickjs_rs
/// restoring into a fresh wasm instance).
#[cfg(feature = "snapshot-linear-mem")]
#[instrument(skip_all, level = "Info")]
fn snapshot_reset(_function_call: FunctionCall) -> Result<Vec<u8>> {
    let engine = CUR_ENGINE.lock();
    let engine = engine
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no engine"))?;
    let linker = CUR_LINKER.lock();
    let linker = linker
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no linker"))?;
    let module = CUR_MODULE.lock();
    let module = module
        .deref()
        .as_ref()
        .ok_or_else(|| snapshot_state_err("no module"))?;
    let mut store = Store::new(engine, ());
    let instance = linker
        .instantiate(&mut store, module)
        .map_err(map_wasmtime_error)?;
    *CUR_STORE.lock() = Some(store);
    *CUR_INSTANCE.lock() = Some(instance);
    Ok(get_flatbuffer_result::<()>(()))
}

// GuestFunctionDefinition expects a function pointer
#[no_mangle]
#[instrument(skip_all, level = "Info")]
pub extern "C" fn hyperlight_main() {
    platform::register_page_fault_handler();

    register_function(GuestFunctionDefinition::new(
        "PrintOutput".to_string(),
        vec![ParameterType::String],
        ReturnType::Int,
        print_output_with_host_print,
    ));

    register_function(GuestFunctionDefinition::new(
        "InitWasmRuntime".to_string(),
        vec![ParameterType::VecBytes],
        ReturnType::Int,
        init_wasm_runtime,
    ));

    register_function(GuestFunctionDefinition::new(
        "LoadWasmModule".to_string(),
        vec![ParameterType::VecBytes, ParameterType::Int],
        ReturnType::Int,
        load_wasm_module,
    ));
    register_function(GuestFunctionDefinition::new(
        "LoadWasmModulePhys".to_string(),
        vec![ParameterType::ULong, ParameterType::ULong],
        ReturnType::Void,
        load_wasm_module_phys,
    ));

    #[cfg(feature = "snapshot-linear-mem")]
    {
        register_function(GuestFunctionDefinition::new(
            "SnapshotMemSize".to_string(),
            vec![],
            ReturnType::ULong,
            snapshot_mem_size,
        ));
        register_function(GuestFunctionDefinition::new(
            "SnapshotReadMem".to_string(),
            vec![ParameterType::ULong, ParameterType::ULong],
            ReturnType::VecBytes,
            snapshot_read_mem,
        ));
        register_function(GuestFunctionDefinition::new(
            "SnapshotGrowMemTo".to_string(),
            vec![ParameterType::ULong],
            ReturnType::Void,
            snapshot_grow_mem_to,
        ));
        register_function(GuestFunctionDefinition::new(
            "SnapshotWriteMem".to_string(),
            vec![ParameterType::ULong, ParameterType::VecBytes],
            ReturnType::Void,
            snapshot_write_mem,
        ));
        register_function(GuestFunctionDefinition::new(
            "SnapshotGetGlobals".to_string(),
            vec![],
            ReturnType::VecBytes,
            snapshot_get_globals,
        ));
        register_function(GuestFunctionDefinition::new(
            "SnapshotSetGlobals".to_string(),
            vec![ParameterType::VecBytes],
            ReturnType::Void,
            snapshot_set_globals,
        ));
        register_function(GuestFunctionDefinition::new(
            "SnapshotReset".to_string(),
            vec![],
            ReturnType::Void,
            snapshot_reset,
        ));
    }
}
