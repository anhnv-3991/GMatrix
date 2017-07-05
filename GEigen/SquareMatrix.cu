#include "SquareMatrix.h"
#include "debug.h"

namespace gpu {
SquareMatrix::SquareMatrix(int size)
{
	rows_ = cols_ = size;
	offset_ = 1;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(float) * rows_ * cols_ * offset_));
}

IdentityMatrix::IdentityMatrix(int size)
{
	rows_ = cols_ = size;
	offset_ = 1;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(float) * rows_ * cols_ * offset_));

	float *host_buff = (float*)malloc(sizeof(float) * rows_ * cols_ * offset_);

	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			host_buff[i * cols_ + j] = (i != j) ? 0 : 1;
		}
	}

	checkCudaErrors(cudaMemcpy(buffer_, host_buff, sizeof(float) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));

	free(host_buff);
}
}
