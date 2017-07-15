#ifndef MATRIX_DEVICE_H_
#define MATRIX_DEVICE_H_

#include "Matrix.h"

namespace gpu {
class MatrixDevice : public Matrix {
public:
	MatrixDevice();
	MatrixDevice(int rows, int cols);

	CUDAH MatrixDevice(int rows, int cols, int offset, double *buffer) {
		rows_ = rows;
		cols_ = cols;
		offset_ = offset;
		buffer_ = buffer;
	}

	CUDAH bool isEmpty() {
		return (rows_ == 0 && cols_ == 0);
	}

};

class SquareMatrixDevice : public MatrixDevice {
public:
	SquareMatrixDevice(int size) : MatrixDevice(size, size) {};
};
}

#endif
