#ifndef MATRIX_HOST_H_
#define MATRIX_HOST_H_

#include "Matrix.h"
#include "MatrixDevice.h"

namespace gpu {
class MatrixHost : public Matrix {
public:
	MatrixHost() : Matrix() {};
	MatrixHost(int rows, int cols);

	bool moveToGpu(MatrixDevice output);
	bool moveToHost(MatrixDevice input);

	bool operator!=(const MatrixHost mat) const;
};

class SquareMatrixHost: public MatrixHost {
public:
	SquareMatrixHost(int size) : MatrixHost(size, size) {};
};

class IdentityMatrixHost: public SquareMatrixHost {
public:
	IdentityMatrixHost(int size);
};
}

#endif
