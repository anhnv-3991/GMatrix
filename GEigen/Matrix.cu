#include "Matrix.h"
#include "debug.h"

namespace gpu {
Matrix::Matrix(int rows, int cols)
{
	rows_ = rows;
	cols_ = cols;
	offset_ = 1;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(float) * rows_ * cols_ * offset_));
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(float) * rows_ * cols_ * offset_));
	checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" __global__ void compareNotEqual(Matrix left, Matrix right, bool *res)
{
	for (int i = 0; i < left.getRowsCount(); i++) {
		for (int j = 0; j < left.getColsCount(); j++) {
			if (left(i, j) != right(i, j)) {
				*res = false;
				return;
			}
		}
	}

	*res = true;
}

bool Matrix::operator!=(const Matrix mat) const
{
	if (rows_ != mat.rows_ || cols_ != mat.cols_)
		return false;

	bool *result;

	checkCudaErrors(cudaMallocHost(&result, sizeof(bool)));
	compareNotEqual<<<1, 1>>>(*this, mat, result);
	checkCudaErrors(cudaDeviceSynchronize());

	bool retval = *result;
	checkCudaErrors(cudaFreeHost(result));

	return retval;
}

bool Matrix::setValFromHost(int row, int col, float val)
{
	if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
		return false;
	
	checkCudaErrors(cudaMemcpy(buffer_ + row * cols_ + col, &val, sizeof(float), cudaMemcpyHostToDevice));

	return true;
}

}
