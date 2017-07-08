#ifndef MATRIX_DEVICE_H_
#define MATRIX_DEVICE_H_

#include "Matrix.h"

namespace gpu {
class MatrixDevice : public Matrix {
public:
	MatrixDevice();
	MatrixDevice(int rows, int cols);

	CUDAH MatrixDevice(int rows, int cols, int offset, float *buffer) {
		rows_ = rows;
		cols_ = cols;
		offset_ = offset;
		buffer_ = buffer;
	}

};

class SquareMatrixDevice : public MatrixDevice {
public:
	SquareMatrixDevice(int size) : MatrixDevice(size, size) {};
};
}

#endif
