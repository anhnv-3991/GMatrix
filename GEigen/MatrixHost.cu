#include "MatrixHost.h"
#include "debug.h"
#include <cstring>

namespace gpu {
	MatrixHost::MatrixHost(int rows, int cols) {
		rows_ = rows;
		cols_ = cols;
		offset_ = 1;

		buffer_ = (float*)malloc(sizeof(float) * rows_ * cols_ * offset_);
		memset(buffer_, 0, sizeof(float) * rows_ * cols_ * offset_);
	}

	bool MatrixHost::operator!=(const MatrixHost mat) const
	{
		if (rows_ != mat.rows_ || cols_ != mat.cols_)
			return false;

		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				if (buffer_[(i * cols_ + j) * offset_] != mat.at(i, j))
					return false;
			}
		}

		return true;
	}

	extern "C" __global__ void copyMatrixDevToDev(MatrixDevice input, MatrixDevice output) {
		int row = threadIdx.x;
		int col = threadIdx.y;
		int rows_num = input.getRowsCount();
		int cols_num = input.getColsCount();

		if (row < rows_num && col < cols_num)
			output(row, col) = input(row, col);
	}

	bool MatrixHost::moveToGpu(MatrixDevice output) {
		if (rows_ != output.getRowsCount() || cols_ != output.getColsCount())
			return false;

		if (offset_ == output.getOffset()) {
			checkCudaErrors(cudaMemcpy(output.getBuffer(), buffer_, sizeof(float) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));
			return true;
		}
		else {
			float *tmp;

			checkCudaErrors(cudaMalloc(&tmp, sizeof(float) * rows_ * cols_ * offset_));
			checkCudaErrors(cudaMemcpy(tmp, buffer_, sizeof(float) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));

			MatrixDevice tmp_output(rows_, cols_, offset_, tmp);

			dim3 block_x(rows_, cols_, 1);
			dim3 grid_x(1, 1, 1);

			copyMatrixDevToDev<<<grid_x, block_x>>>(tmp_output, output);
			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaFree(tmp));

			return true;
		}
	}

	bool MatrixHost::moveToHost(MatrixDevice input) {
		if (rows_ != input.getRowsCount() || cols_ != input.getColsCount())
			return false;

		if (offset_ == input.getOffset()) {
			checkCudaErrors(cudaMemcpy(buffer_, input.getBuffer(), sizeof(float) * rows_ * cols_ * offset_, cudaMemcpyDeviceToHost));
			return true;
		}
		else {
			float *tmp;

			checkCudaErrors(cudaMalloc(&tmp, sizeof(float) * rows_ * cols_ * offset_));

			MatrixDevice tmp_output(rows_, cols_, offset_, tmp);

			dim3 block_x(rows_, cols_, 1);
			dim3 grid_x(1, 1, 1);

			copyMatrixDevToDev << <grid_x, block_x >> >(input, tmp_output);
			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaMemcpy(buffer_, tmp, sizeof(float) * rows_ * cols_ * offset_, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(tmp));

			return true;
		}
	}
}