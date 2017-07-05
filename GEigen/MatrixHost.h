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

		//Host-side methods
		bool operator!=(const MatrixHost mat) const;

	private:
	};
}

#endif
