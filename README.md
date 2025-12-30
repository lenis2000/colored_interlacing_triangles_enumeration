# Enumeration of Colored Interlacing Triangles

Code for enumerating colored interlacing n-triangles T_N(n).

## Quick Start

```bash
make                    # Build all available targets
./enumerate 4 3         # T_3(4) = 191,232 (CPU)
./enumerate_metal 4 3   # Same, using GPU
make help               # Show all options
```

## Files

| File | Description |
|------|-------------|
| `enumerate_general.cpp` | CPU enumeration (any N, n) |
| `enumerate_metal.cpp` | Metal GPU + disk-chunked (any N, n) |

## Build System

Run `make help` for full documentation. Auto-detects:

| Platform | OpenMP | Metal |
|----------|--------|-------|
| macOS Apple Silicon | Via `brew install gcc` | Yes |
| macOS Intel | Via `brew install gcc` | No |
| Linux | Native | No |

### Metal GPU (`enumerate_metal`)
- GPU-accelerated interlacing checks
- Memory chunking: 500MB GPU batches
- Disk chunking: For levels >100M states, 50M-state chunks with checkpoint/resume
- Falls back to CPU if Metal unavailable

### CPU (`enumerate`)
- Recursive level-by-level construction
- OpenMP parallelization
