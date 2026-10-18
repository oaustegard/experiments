Seconds inside the program around `run(n)`, median of reps; (Nx) = speed vs native CPython 3.11.

| arm | nbody | mandelbrot | matmul | dictwords |
|---|---|---|---|---|
| c-native | 0.072s (22.66x) | 0.026s (32.55x) | 0.010s (50.69x) | 0.016s (43.10x) |
| c-ceiling [wasmtime] | 0.125s (13.18x) | 0.027s (31.98x) | 0.012s (41.90x) | 0.031s (22.36x) |
| c-ceiling [node] | 0.099s (16.63x) | 0.033s (26.31x) | 0.018s (27.33x) | 0.034s (20.53x) |
| spy [wasmtime] | 0.181s (9.06x) | 0.030s (29.00x) | 0.111s (4.46x) | 0.368s (1.90x) |
| mypyc-wasi [wasmtime] | 2.665s (0.62x) | 0.042s (20.33x) | 0.643s (0.77x) | 1.623s (0.43x) |
| mypyc-wasi [node] | 2.136s (0.77x) | 0.044s (19.51x) | 0.693s (0.72x) | 1.640s (0.43x) |
| cython-wasi [wasmtime] | 1.910s (0.86x) | 0.181s (4.74x) | 1.417s (0.35x) | 1.691s (0.41x) |
| cython-wasi [node] | 1.784s (0.92x) | 0.201s (4.25x) | 1.155s (0.43x) | 1.430s (0.49x) |
| cpython-native-3.11 | 1.642s (1.00x) | 0.856s (1.00x) | 0.497s (1.00x) | 0.697s (1.00x) |
| cpython-native-3.13 | 1.626s (1.01x) | 0.942s (0.91x) | 0.527s (0.94x) | 0.782s (0.89x) |
| py2wasm [node] | 3.379s (0.49x) | 1.489s (0.57x) | 1.356s (0.37x) | 1.926s (0.36x) |
| py2wasm [wasmtime] | 3.685s (0.45x) | 1.488s (0.58x) | 1.468s (0.34x) | 2.140s (0.33x) |
| cpython-wasi [wasmtime] | 3.800s (0.43x) | 1.883s (0.45x) | 1.254s (0.40x) | 2.335s (0.30x) |
| cpython-wasi [node] | 4.867s (0.34x) | 2.703s (0.32x) | 1.769s (0.28x) | 2.056s (0.34x) |

Speed vs cpython-wasi under wasmtime (the in-wasm interpreter baseline):

| arm | nbody | mandelbrot | matmul | dictwords |
|---|---|---|---|---|
| c-native | 52.44x | 71.58x | 127.85x | 144.34x |
| c-ceiling [wasmtime] | 30.48x | 70.33x | 105.68x | 74.87x |
| c-ceiling [node] | 38.48x | 57.84x | 68.92x | 68.75x |
| spy [wasmtime] | 20.97x | 63.76x | 11.25x | 6.35x |
| mypyc-wasi [wasmtime] | 1.43x | 44.71x | 1.95x | 1.44x |
| mypyc-wasi [node] | 1.78x | 42.90x | 1.81x | 1.42x |
| cython-wasi [wasmtime] | 1.99x | 10.42x | 0.88x | 1.38x |
| cython-wasi [node] | 2.13x | 9.35x | 1.09x | 1.63x |
| cpython-native-3.11 | 2.31x | 2.20x | 2.52x | 3.35x |
| cpython-native-3.13 | 2.34x | 2.00x | 2.38x | 2.99x |
| py2wasm [node] | 1.12x | 1.26x | 0.92x | 1.21x |
| py2wasm [wasmtime] | 1.03x | 1.26x | 0.85x | 1.09x |
| cpython-wasi [wasmtime] | 1.00x | 1.00x | 1.00x | 1.00x |
| cpython-wasi [node] | 0.78x | 0.70x | 0.71x | 1.14x |

Shipped bytes per kernel (artifact.txt):

| arm | nbody | mandelbrot | matmul | dictwords |
|---|---|---|---|---|
| c-native | 0.02 MB | 0.02 MB | 0.02 MB | 0.02 MB |
| c-ceiling [wasmtime] | 0.04 MB | 0.03 MB | 0.03 MB | 0.04 MB |
| c-ceiling [node] | 0.04 MB | 0.03 MB | 0.03 MB | 0.04 MB |
| spy [wasmtime] | 1.07 MB | 1.03 MB | 1.06 MB | 1.10 MB |
| mypyc-wasi [wasmtime] | 6.86 MB | 6.83 MB | 6.84 MB | 6.85 MB |
| mypyc-wasi [node] | 6.86 MB | 6.83 MB | 6.84 MB | 6.85 MB |
| cython-wasi [wasmtime] | 6.88 MB | 6.84 MB | 6.85 MB | 6.85 MB |
| cython-wasi [node] | 6.88 MB | 6.84 MB | 6.85 MB | 6.85 MB |
| py2wasm [node] | 27.07 MB | 27.03 MB | 27.05 MB | 27.05 MB |
| py2wasm [wasmtime] | 27.07 MB | 27.03 MB | 27.05 MB | 27.05 MB |
| cpython-wasi [wasmtime] | 6.80 MB | 6.80 MB | 6.80 MB | 6.80 MB |
| cpython-wasi [node] | 6.80 MB | 6.80 MB | 6.80 MB | 6.80 MB |
