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

//! Round-trip test for the generic linear-memory + globals snapshot guest
//! functions exposed by the `snapshot-linear-mem` feature.
//!
//! This drives the `Snapshot*` guest functions over the public
//! `call_guest_function` seam to prove that a loaded guest's resumable state
//! (linear memory + mutable globals) can be captured and faithfully restored
//! into a freshly re-instantiated module, without a whole-VM image.
//!
//! Prerequisites to RUN (not to compile): the `RunWasm.aot` example module must
//! be built into `x64/{debug,release}/` (`just build-wasm-examples
//! build-rust-wasm-examples`) and the host must have hardware virtualization
//! available (Linux/KVM, or Windows/WHP). The whole file is gated behind the
//! `snapshot-linear-mem` Cargo feature, so without it this is an empty test
//! binary.
#![cfg(feature = "snapshot-linear-mem")]

use examples_common::get_wasm_module_path;
use hyperlight_wasm::{HyperlightError, LoadedWasmSandbox, Result, SandboxBuilder};

/// Chunk size for streaming linear memory across the guest ABI. Kept well below
/// the configured input/output buffers so each read/write fits in one call.
const CHUNK: u64 = 64 * 1024;
const IO_BUFFER: usize = 256 * 1024;

fn get_time_since_boot_microsecond() -> Result<i64> {
    let res = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .expect("time went backwards")
        .as_micros();
    i64::try_from(res).map_err(HyperlightError::IntConversionFailure)
}

fn test_host_func(a: i32) -> i32 {
    a
}

fn load() -> LoadedWasmSandbox {
    let mut sandbox = SandboxBuilder::new()
        .with_guest_input_buffer_size(IO_BUFFER)
        .with_guest_output_buffer_size(IO_BUFFER)
        .build()
        .unwrap();
    sandbox
        .register(
            "GetTimeSinceBootMicrosecond",
            get_time_since_boot_microsecond,
        )
        .unwrap();
    sandbox.register("TestHostFunc", test_host_func).unwrap();
    let wasm_sandbox = sandbox.load_runtime().unwrap();
    // Defaults to the canonical RunWasm.aot example; override to point at any
    // other AOT module that exports `memory` and an `add(i32, i32) -> i32`.
    let module =
        std::env::var("HL_SNAPSHOT_TEST_MODULE").unwrap_or_else(|_| "RunWasm.aot".to_string());
    let mod_path = get_wasm_module_path(&module).unwrap();
    wasm_sandbox.load_module(mod_path).unwrap()
}

fn mem_size(sb: &mut LoadedWasmSandbox) -> u64 {
    sb.call_guest_function("SnapshotMemSize", ()).unwrap()
}

fn read_full_memory(sb: &mut LoadedWasmSandbox) -> Vec<u8> {
    let size = mem_size(sb);
    let mut out = Vec::with_capacity(size as usize);
    let mut off = 0u64;
    while off < size {
        let len = CHUNK.min(size - off);
        let chunk: Vec<u8> = sb
            .call_guest_function("SnapshotReadMem", (off, len))
            .unwrap();
        assert_eq!(chunk.len() as u64, len, "short read at offset {off}");
        out.extend_from_slice(&chunk);
        off += len;
    }
    out
}

fn write_full_memory(sb: &mut LoadedWasmSandbox, image: &[u8]) {
    let mut off = 0usize;
    while off < image.len() {
        let end = (off + CHUNK as usize).min(image.len());
        let chunk = image[off..end].to_vec();
        sb.call_guest_function::<()>("SnapshotWriteMem", (off as u64, chunk))
            .unwrap();
        off = end;
    }
}

/// Captures linear memory + mutable globals, deliberately mutates and grows the
/// instance, resets it, restores the captured state, and asserts byte-exact
/// reconstruction plus a working post-restore guest call.
#[test]
fn snapshot_restore_round_trip() {
    let mut sb = load();

    // Confirm the guest is callable and capture a behavioral baseline.
    let baseline_add: i32 = sb.call_guest_function("add", (2i32, 3i32)).unwrap();

    // ---- Capture S0 ----
    let size0 = mem_size(&mut sb);
    assert!(size0 >= 65536, "expected at least one page, got {size0}");
    let image0 = read_full_memory(&mut sb);
    let globals0: Vec<u8> = sb.call_guest_function("SnapshotGetGlobals", ()).unwrap();
    assert_eq!(image0.len() as u64, size0);

    // ---- Mutation proves Read/Write observe state changes ----
    let off = size0 - 64;
    let orig: Vec<u8> = sb
        .call_guest_function("SnapshotReadMem", (off, 16u64))
        .unwrap();
    let mutated: Vec<u8> = orig.iter().map(|b| b ^ 0xA5).collect();
    assert_ne!(orig, mutated);
    sb.call_guest_function::<()>("SnapshotWriteMem", (off, mutated.clone()))
        .unwrap();
    let read_back: Vec<u8> = sb
        .call_guest_function("SnapshotReadMem", (off, 16u64))
        .unwrap();
    assert_eq!(read_back, mutated, "write+read did not round-trip");

    // ---- Grow proves SnapshotGrowMemTo ----
    let grow_to = size0 + 65536;
    sb.call_guest_function::<()>("SnapshotGrowMemTo", (grow_to,))
        .unwrap();
    assert_eq!(
        mem_size(&mut sb),
        grow_to,
        "memory did not grow as requested"
    );

    // ---- Reset re-instantiates the module in a fresh store ----
    sb.call_guest_function::<()>("SnapshotReset", ()).unwrap();

    // ---- Restore S0: grow back, rewrite the image, set globals ----
    sb.call_guest_function::<()>("SnapshotGrowMemTo", (size0,))
        .unwrap();
    assert_eq!(mem_size(&mut sb), size0);
    write_full_memory(&mut sb, &image0);
    sb.call_guest_function::<()>("SnapshotSetGlobals", (globals0.clone(),))
        .unwrap();

    // ---- Verify byte-exact reconstruction ----
    let image2 = read_full_memory(&mut sb);
    assert_eq!(image2, image0, "linear memory was not faithfully restored");
    let globals2: Vec<u8> = sb.call_guest_function("SnapshotGetGlobals", ()).unwrap();
    assert_eq!(globals2, globals0, "globals were not faithfully restored");

    // ---- Restored instance is functional (time-travel) ----
    let restored_add: i32 = sb.call_guest_function("add", (2i32, 3i32)).unwrap();
    assert_eq!(
        restored_add, baseline_add,
        "restored instance does not behave like the original"
    );
}
