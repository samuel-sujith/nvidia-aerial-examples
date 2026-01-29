## Canonical Module API Example

This example shows the expected `IModule` lifecycle in the NVIDIA Aerial Framework,
without using `IPipeline`.

### Lifecycle Steps
- `get_requirements()` — query memory requirements
- `setup_memory()` — provide a `ModuleMemorySlice`
- `set_inputs()` — bind external device buffers using `PortInfo`
- `configure_io()` — configure the module for execution
- `warmup()` — prepare kernels/graphs
- `execute()` — run the module on a CUDA stream
- `get_outputs()` — read output device pointers

### Files
- `canonical_module_api_example.cpp` — standalone example with explicit memory setup
- `CMakeLists.txt` — build rules for this example

### Build
From the repository root:

```
cmake -S . -B build \
  -DAERIAL_FRAMEWORK_ROOT=/path/to/aerial-framework/install
cmake --build build -j
```

### Run
```
./build/canonical_module_api/canonical_module_api_example
```

### What It Demonstrates
- Module-level execution without `IPipeline`
- Manual memory slice allocation
- Device I/O wiring via `PortInfo`
