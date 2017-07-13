#ifndef GPU_COMMON_H_
#define GPU_COMMON_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDAH __forceinline__ __host__ __device__
#define BLOCK_SIZE_X 1024

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_Z 4
#endif
