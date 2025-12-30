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

## Benchmark Results (Metal GPU)

| (n, N) | T_N(n) | Time |
|--------|--------|------|
| (3, 3) | 528 | <1ms |
| (3, 4) | 8,160 | 10ms |
| (4, 3) | 191,232 | 10ms |
| (3, 5) | 179,520 | 75ms |
| (5, 3) | 257,794,560 | ~min |
| (6, 3) | 1,012,737,392,640 | ~hours (disk-chunked) |

## Algorithm

Both use S_n symmetry: enumerate **canonical** states (first=1, last=n), multiply by n!.

This reduces state space dramatically:
- n=4, level 3: 25,200 canonical vs 369,600 full states

### Metal GPU (`enumerate_metal`)
- GPU-accelerated interlacing checks
- Memory chunking: 500MB GPU batches
- Disk chunking: For levels >100M states, 50M-state chunks with checkpoint/resume
- Falls back to CPU if Metal unavailable

### CPU (`enumerate`)
- Recursive level-by-level construction
- OpenMP parallelization
