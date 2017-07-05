#ifndef GVECTOR_H_
#define GVECTOR_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.h"
#include "common.h"

namespace gpu {

	class Vector : protected Matrix {
	public:
		CUDAH Vector() {
			rows_ = cols_ = offset_ = 0;
			buffer_ = NULL;
		}

		CUDAH Vector(int size, int offset, float *buffer) {
			rows_ = 1;
			cols_ = size;
			offset_ = offset;
			buffer_ = buffer;
		}

		CUDAH float& operator()(int index) {
			if (index < 0 || index >= cols_)
				exit(EXIT_FAILURE);

			return buffer_[index * offset_];
		}

		CUDAH float at(int index) const {
			if (index < 0 || index >= cols_)
				exit(EXIT_FAILURE);

			return buffer_[index * offset_];
		}

		CUDAH float operator*(const Vector input) const {
			float retval = 0;

			for (int i = 0; i < cols_; i++) {
				retval += buffer_[i * offset_] * input.at(i);
			}

			return retval;
		}
	};

}

#endif
