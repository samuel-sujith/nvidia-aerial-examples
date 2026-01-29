## Canonical IPipeline Example

This example shows the expected `IPipeline` lifecycle in the NVIDIA Aerial Framework.

### Lifecycle Steps
- `setup()` — allocate and configure pipeline resources
- `configure_io()` — bind external device buffers using `PortInfo`
- `warmup()` — prepare kernels/graphs for execution
- `execute_stream()` — run the pipeline on a CUDA stream

### Files
- `canonical_ipipeline_example.cpp` — standalone example with explicit I/O setup
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
./build/canonical_ipipeline/canonical_ipipeline_example
```

### What It Demonstrates
- Creating an `IPipeline` instance
- Creating and wiring `PortInfo` inputs/outputs
- Using `DynamicParams` (empty here)
- Correct ordering of `setup → configure_io → warmup → execute_stream`
