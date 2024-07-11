# Optimized 2D Convolution for Bitmap Images

This repository contains an advanced implementation of 2D convolution for processing bitmap images. It leverages modern C++ features, AVX2 intrinsics, OpenMP for parallelization, and efficient memory management techniques to significantly enhance performance over traditional methods.

## Features

- **AVX2 Intrinsics**: Utilizes AVX2 vector instructions for efficient floating-point arithmetic, enabling the processing of multiple pixels simultaneously.
- **OpenMP Parallelization**: Employs OpenMP to distribute computations across multiple threads, maximizing the utilization of multi-core processors.
- **Memory Mapping**: Utilizes `mmap` for efficient file I/O and memory management, reducing overhead and improving performance.
- **Custom Task Scheduling**: Implements a custom task scheduling mechanism to ensure balanced workload distribution among threads, optimizing resource usage.
- **Edge Handling**: Efficiently manages edge cases where the convolution kernel overlaps the image boundaries, preventing errors and maintaining accuracy.

## Dependencies

- **GCC**: Requires GCC with support for AVX2 and OpenMP.
- **OpenMP**: Essential for parallelization.
- **Pthreads**: Used for setting thread affinity.
- **C++17 Filesystem Library**: For file path management.

## Compilation

To compile the code, ensure you have the necessary dependencies installed and use the following command:

```
g++ -O3 -march=native -fopenmp -o optimized_convolution src/optimized_convolution.cpp
```


## Usage

To use the optimized convolution implementation, execute the provided `runnerscript.sh` script from the parent directory of the `src` directory. This script compiles the `optimized_convolution.cpp` file and executes the compiled program with the necessary parameters.

Example:

```
 ./runnerscript.sh
 ```
 
 
 
Ensure that the `bitmap_path`, `kernel`, `num_rows`, and `num_cols` variables are correctly set within the script or the program.

## Performance

The optimization techniques employed in this implementation lead to a performance improvement of approximately 60x compared to a naive 2D convolution approach. Key optimizations include:

- **Vectorization**: Processes 8 pixels in parallel using AVX2 intrinsics.
- **Parallelization**: Distributes the workload across 24 threads using OpenMP.
- **Efficient Memory Access**: Minimizes cache misses and improves memory access patterns.

## Notes

- Verify that your CPU supports AVX2 instructions by running `lscpu | grep avx2` on a Linux system.
- Adjust the number of threads based on your system's capabilities to achieve optimal performance.

