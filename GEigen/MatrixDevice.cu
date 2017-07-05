#include "MatrixDevice.h"
#include "debug.h"

namespace gpu {
	MatrixDevice::MatrixDevice() {
		rows_ = cols_ = offset_ = 0;
		buffer_ = NULL;
	}

	MatrixDevice::MatrixDevice(int rows, int cols) {
		rows_ = rows;
		cols_ = cols;
		offset_ = 1;

		checkCudaErrors(cudaMalloc(&buffer_, sizeof(float) * rows_ * cols_ * offset_));
		checkCudaErrors(cudaMemset(buffer_, 0, sizeof(float) * rows_ * cols_ * offset_));
		checkCudaErrors(cudaDeviceSynchronize());
	}
}