#ifndef GMATRIX_H_
#define GMATRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <float.h>

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

	CUDAH double *buffer() const {
		return buffer_;
	}

	CUDAH void setRows(int rows) { rows_ = rows; }
	CUDAH void setCols(int cols) { cols_ = cols; }
	CUDAH void setOffset(int offset) { offset_ = offset; }
	CUDAH void setBuffer(double *buffer) { buffer_ = buffer; }

	//Need to fix. Only reducing rows is OK now.
	CUDAH void resize(int rows, int cols) {
		rows_ = rows;
		cols_ = cols;
	}

	CUDAH double *cellAddr(int row, int col) {
		if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
			return NULL;

		return buffer_ + (row * cols_ + col) * offset_;
	}

	CUDAH double *cellAddr(int index) {
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

	CUDAH double& operator()(int row, int col) {
		return buffer_[(row * cols_ + col) * offset_];
	}

	CUDAH double& operator()(int index) {
		if (rows_ == 1) {
			if (index >= 0 && index < cols_)
				return buffer_[index * offset_];
		} else if (cols_ == 1) {
			if (index >= 0 && index < rows_)
				return buffer_[index * offset_];
		}
		double val = DBL_MAX;

		return val;
	}

	CUDAH double at(int row, int col) const {
		if (row < 0 || col < 0 || row >= rows_ || col >= cols_)
			exit(EXIT_FAILURE);

		return buffer_[(row * cols_ + col) * offset_];
	}

	CUDAH bool operator*=(double val) {
		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				buffer_[(i * cols_ + j) * offset_] *= val;
			}
		}

		return true;
	}

	CUDAH bool operator/=(double val) {
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
	double *buffer_;
	int rows_, cols_, offset_;
};

}

#endif


