#include "Matrix.h"
#include "debug.h"

namespace gpu {
Matrix::Matrix(int rows, int cols)
{
	rows_ = rows;
	cols_ = cols;
	offset_ = 1;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(float) * rows_ * cols_ * offset_));
}
}
