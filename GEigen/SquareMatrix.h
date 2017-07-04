#ifndef GSQUARE_MATRIX_H_
#define GSQUARE_MATRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.h"
#include "common.h"

namespace gpu {

	class SquareMatrix : public Matrix {
	public:
		CUDAH SquareMatrix() {
			rows_ = cols_ = 0;
			offset_ = 0;
			buffer_ = NULL;
		}

		SquareMatrix(int size);

		CUDAH SquareMatrix(int size, int offset, float *buffer) {
			rows_ = cols_ = size;
			offset_ = offset;
			buffer_ = buffer;
		}

		CUDAH float det(float *temp_buffer) {
			if (rows_ != cols_)
				exit(EXIT_FAILURE);

			if (rows_ == 1)
				return buffer_[0];

			if (rows_ == 2)
				return (buffer_[0] * buffer_[3 * offset_] - buffer_[offset_] * buffer_[2 * offset_]);

			if (rows_ == 3)
				return (buffer_[0] * buffer_[4 * offset_] * buffer_[8 * offset_] + buffer_[offset_] * buffer_[5 * offset_] * buffer_[6 * offset_]
					+ buffer_[2 * offset_] * buffer_[3 * offset_] * buffer_[7 * offset_] - buffer_[2 * offset_] * buffer_[4 * offset_] * buffer_[6 * offset_]
					- buffer_[offset_] * buffer_[3 * offset_] * buffer_[8 * offset_] - buffer_[0] * buffer_[5 * offset_] * buffer_[7 * offset_]);

			float retval = 0;
			float sign = 1;

			for (int i = 0; i < rows_; i++, sign *= (-1)) {
				SquareMatrix sub_matrix(rows_ - 1, offset_, temp_buffer);

				getSubMatrix(0, i, &sub_matrix);

				retval += sign * sub_matrix.det(temp_buffer + (rows_ - 1) * (rows_ - 1) * offset_);
			}

			return retval;
		}

		CUDAH bool inverse(float *temp_buffer, Matrix *output) {
			if (rows_ != cols_)
				return false;

			float det_val = det(temp_buffer);

			if (det_val)
				return false;

			if (rows_ == 1) {
				output->set(0, 0, 1 / det_val);

				return true;
			}

			float sign = 1;

			for (int i = 0; i < rows_; i++) {
				for (int j = 0; j < cols_; j++, sign *= (-1)) {
					SquareMatrix sub_matrix(rows_ - 1, offset_, temp_buffer);

					getSubMatrix(i, j, &sub_matrix);

					output->set(i, j, sign * 1 / det_val * sub_matrix.det(temp_buffer + (rows_ - 1) * (cols_ - 1) * offset_));
				}
			}

			return true;
		}
	};

	class IdentityMatrix : public SquareMatrix {
	public:
		CUDAH IdentityMatrix() {
			rows_ = cols_ = 0;
			offset_ = 0;
			buffer_ = NULL;
		}

		IdentityMatrix(int size);

		CUDAH IdentityMatrix(int size, int offset, float *buffer) {
			rows_ = size;
			cols_ = size;
			offset_ = offset;
			buffer_ = buffer;

			for (int i = 0; i < rows_; i++) {
				for (int j = 0; j < cols_; j++) {
					buffer[(i * cols_ + j) * offset_] = (i == j) ? 1 : 0;
				}
			}
		}
	};
}

#endif
