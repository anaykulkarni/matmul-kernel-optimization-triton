We create a Triton-based implementation of a matrix multiplication + ReLU + add kernel. The kernel computes the matrix function **D = ReLU(A × B + C)** where **A** is of shape *(M, K)*, **B** is of shape *(K, N)*, and **C & D** are of shape *(M, N)*. We will break the kernel down into four main steps:

1. **Tile Assignment**  
2. **Shared Memory Tiling + Cooperative Fetching**  
3. **Register Tiling (Accumulator)**  
4. **Operator Fusion**
5. **Write Cache/Epilogue (Store Results)**

Each section below explains the purpose and the implementation details of each step.

### Setup
### 1.1 Tile Assignment
**Tile Assignment** is the process by which the overall matrix **C** is divided into smaller submatrices (tiles). Each kernel instance (a GPU “program” or thread block) is responsible for computing one tile of **C**. This division allows the computation to be parallelized across many threads.

Each kernel launch instance (denoted by a unique program id `pid`) should be mapped to a specific tile in **C**. The tile is defined by two indices: one for the row (denoted `pid_m`) and one for the column (`pid_n`).

### 1.2 Shared Memory Tiling + Cooperative Fetching
**Shared Memory Tiling** is used to load sub-tiles (smaller blocks) of matrices A and B into faster, on-chip memory. Threads within a kernel instance load these sub-tiles, reducing the number of global memory accesses.

Some things to keep in mind:
- You may need to use `tl.arange` to help compute offsets for the rows and columns.
- Use masks to make sure that out-of-bound memory accesses do not occur.

### 1.3 Register Tiling (Accumulator)
**Register Tiling** is the use of registers to hold intermediate results (partial sums) during the matrix multiplication. For this section, you will need to use an accumulator (a BLOCK_SIZE_M by BLOCK_SIZE_N matrix initialized to zeros) to accumulate results of dot products computed over tiles.

After accumulation you can optionally choose to fuse an activation function (like leaky ReLU) - this is used in practice to make architectures that use lots of matmuls and activation functions together (like transformers) much much faster!

### 1.4 Add and ReLU Fusion
In this step, we fuse the element-wise addition of matrix C and the final ReLU activation directly into the matmul kernel to optimize performance by reducing memory traffic and kernel launch overhead. After computing the matrix multiplication and storing the results in the accumulator, you must load the corresponding tile from matrix C and add it element-wise to the activated accumulator. Then apply the ReLU function using an element-wise maximum operation to set all negative values to zero. 

This fusion of operations avoids writing intermediate results back to global memory and then reloading them for further processing, minimizing latency and making more efficient use of the GPU’s memory hierarchy.

### 1.5 Write Cache/Epilogue
In this step, we write the tile of C back to global memory. This final step ensures that the computed results are stored in the correct locations in the output matrix C. Be sure to also use a mask to prevent invalid indices from being written to. (Use `tl.store` to store your tiles.)

### 1.6 Grid Search
To achieve full credit on part 1, you will have to perform grid search or manually find values for the hyperparameter block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`). We have provided parameters that should result in you achieving around 0.9-1x speedup, but you will have to search for different block sizes to achieve better speedup.