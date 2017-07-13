#ifndef GMATRIX_H_
#define GMATRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

namespace gpu {

class Matrix {
public:
	CUDAH Matrix() {
		buffer_ = NULL;
		rows_ = cols_ = offset_ = 0;
	}
	
	CUDAH int rows() const {
		return rows_;
	}

	CUDAH int cols() const {
		return cols_;
	}

	CUDAH int offset() const {
		return offset_;
	}

	CUDAH float *buffer() const {
		return buffer_;
	}

	CUDAH void setRows(int rows) { rows_ = rows; }
	CUDAH void setCols(int cols) { cols_ = cols; }
	CUDAH void setOffset(int offset) { offset_ = offset; }
	CUDAH void setBuffer(float *buffer) { buffer_ = buffer; }

	//Need to fix. Only reducing rows is OK now.
	CUDAH void resize(int rows, int cols) {
		rows_ = rows;
		cols_ = cols;
	}

	CUDAH float *cellAddr(int row, int col) {
		if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
			return NULL;

		return buffer_ + (row * cols_ + col) * offset_;
	}

	CUDAH float *cellAddr(int index) {
		if (rows_ == 1 && index >= 0 && index < cols_) {
				return buffer_ + index * offset_;
		} 
		else if (cols_ == 1 && index >= 0 && index < rows_) {
				return buffer_ + index * offset_;
		}

		return NULL;
	}

	//Assignment operator
	CUDAH void operator=(const Matrix input) {
		if (rows_ != input.rows_ || cols_ != input.cols_)
			return;

		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				buffer_[(i * cols_ + j) * offset_] = input.at(i, j);
			}
		}
	}

	CUDAH float& operator()(int row, int col) {
		if (row < 0 || col < 0 || row >= rows_ || col >= cols_)
			exit(EXIT_FAILURE);

		return buffer_[(row * cols_ + col) * offset_];
	}

	CUDAH float& operator()(int index) {
		if (rows_ == 1) {
			if (index >= 0 && index < cols_)
				return buffer_[index * offset_];
		} else if (cols_ == 1) {
			if (index >= 0 && index < rows_)
				return buffer_[index * offset_];
		} 

		exit(EXIT_FAILURE);
	}

	CUDAH bool operator==(const Matrix mat) {
		if (rows_ != mat.rows_ || cols_ != mat.cols_)
			return false;

		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				if (at(i, j) != mat.at(i, j))
					return false;
			}
		}

		return true;
	}

	CUDAH bool operator!=(const Matrix mat) {
		if (rows_ == mat.rows_ && cols_ == mat.cols_) {
			for (int i = 0; i < rows_; i++) {
				for (int j = 0; j < cols_; j++)
					if (at(i, j) != mat.at(i, j))
						return true;
			}

			return false;
		}
		else
			return true;
	}

	CUDAH float at(int row, int col) const {
		if (row < 0 || col < 0 || row >= rows_ || col >= cols_)
			exit(EXIT_FAILURE);

		return buffer_[(row * cols_ + col) * offset_];
	}

	/* Get sub Matrix by removing row row and column col
	* from the current matrix.
	*/
	CUDAH bool getSubMatrix(int row, int col, Matrix &output) {
		if (row >= rows_ || row < 0 || col >= cols_ || col < 0)
			return false;

		//Get upper left quater
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				output(i, j) = buffer_[(i * cols_ + j) * offset_];
			}
		}

		//Get upper righ quater
		for (int i = 0; i < row; i++) {
			for (int j = col + 1; j < cols_; j++) {
				output(i, j - 1) = buffer_[(i * cols_ + j) * offset_];
			}
		}

		//Get lower left quater
		for (int i = row + 1; i < rows_; i++) {
			for (int j = 0; j < row; j++) {
				output(i - 1, j) = buffer_[(i * cols_ + j) * offset_];
			}
		}

		//Get lower right quater
		for (int i = row + 1; i < rows_; i++) {
			for (int j = col + 1; j < cols_; j++) {
				output(i - 1, j - 1) = buffer_[(i * cols_ + j) * offset_];
			}
		}

		return true;
	}

	static CUDAH bool add(const Matrix input0, const Matrix input1, Matrix& output) {
		if (input0.rows_ != input1.rows_ || input0.rows_ != output.rows_ || input0.cols_ != input1.cols_ || input0.cols_ != output.cols_)
			return false;

		for (int i = 0; i < input0.rows_; i++) {
			for (int j = 0; j < input0.cols_; j++) {
				output(i, j) = input0.at(i, j) + input1.at(i, j);
			}
		}

		return true;
	}

	static CUDAH bool subtract(const Matrix input0, const Matrix input1, Matrix &output) {
		if (input0.rows_ != input1.rows_ || input0.rows_ != output.rows_ || input0.cols_ != input1.cols_ || input0.cols_ != output.cols_)
			return false;

		for (int i = 0; i < input0.rows_; i++) {
			for (int j = 0; j < input0.cols_; j++) {
				output(i, j) = input0.at(i, j) - input1.at(i, j);
			}
		}

		return true;
	}

	static CUDAH bool multiply(const Matrix input0, const Matrix input1, Matrix &output) {
		if (input0.cols_ != input1.rows_ || input0.rows_ != output.rows_ || input1.cols_ != output.cols_)
			return false;

		for (int i = 0; i < output.rows_; i++) {
			for (int j = 0; j < output.cols_; j++) {
				float tmp = 0;
				for (int k = 0; k < input0.cols_; k++) {
					tmp += input0.at(i, k) * input1.at(k, j);
				}

				output(i, j) = tmp;
			}
		}

		return true;
	}

	CUDAH bool scalarMultiply(float val, Matrix &output) {
		if (rows_ != output.rows_ || cols_ != output.cols_)
			return false;

		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				output(i, j) = buffer_[(i * cols_ + j) * offset_] * val;
			}
		}

		return true;
	}

	CUDAH bool scalarDivide(float val, Matrix &output) {
		if (rows_ != output.rows_ || cols_ != output.cols_ || val == 0)
			return false;

		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				output(i, j) = buffer_[(i * cols_ + j) * offset_] / val;
			}
		}

		return true;
	}

	CUDAH bool operator*=(float val) {
		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				buffer_[(i * cols_ + j) * offset_] *= val;
			}
		}

		return true;
	}

	CUDAH bool operator/=(float val) {
		if (val == 0)
			return false;

		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				buffer_[(i * cols_ + j) * offset_] /= val;
			}
		}

		return true;
	}

	CUDAH bool transpose(Matrix &output) {
		if (rows_ != output.cols_ || cols_ != output.rows_)
			return false;

		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				output(j, i) = buffer_[(i * cols_ + j) * offset_];
			}
		}

		return true;
	}

protected:
	float *buffer_;
	int rows_, cols_, offset_;
};

}

#endif


