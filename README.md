# Graph Random Walk Benchmark

Benchmark for NumPy, PyTorch and C++ random walks

## Methods

-  **NumPy**: Pure python loop (but can handle edge weights)
-  **PyTorch Geometric**: We only use CPU kerne of Py-Torch and GPU is probably even faster
-  **C++ with OpenMP**: C++ code compiled using `torch.utils.cpp_extension`. Uses OpenMP for parallelism. The C++ code is stored in `random_walks.cpp`

## Installation

You probably need some form of cpp compiler (gcc, g++,...). `uv` can handle the rest of the dependencies.
Sync the dependences with `uv sync` and then rund the ebnchmark with `uv run benchmark_random_walk.py`.

The first run will take a few seconds longer as it compiles the C++ extension in the background.

## Notes

I uncommented the Numpy Version ebcause it is very slow. You can play around with the parameters (walk_length, ple, num_walks_per_node,...). PyTorch seems to be faster whith shorter walks and more walks per node. C++ code seems to be faster with longer wllk length. But i havn't done extensive experiments.
