#ifndef GMATRIX_H_
#define GMATRIX_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

namespace gpu {

class Matrix {
public:
	CUDAH Matrix() {
		buffer_ = NULL;
		rows_ = cols_ = offset_ = 0;
	}

	Matrix(int rows, int cols);

	//Device-side methods
	CUDAH int getRowsCount() const {
		return rows_;
	}

	CUDAH int getColsCount() const {
		return cols_;
	}

	CUDAH int getOffset() const {
		return offset_;
	}

	CUDAH float *getBuffer() const {
		return buffer_;
	}

	CUDAH Matrix(int rows, int cols, int offset, float *buffer) {
		rows_ = rows;
		cols_ = cols;
		offset_ = offset;
		buffer_ = buffer;
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

	CUDAH float at(int row, int col) const {
		if (row < 0 || col < 0 || row >= rows_ || col >= cols_)
			exit(EXIT_FAILURE);

		return buffer_[(row * cols_ + col) * offset_];
	}

	CUDAH bool set(int row, int col, float val) {
		if (row < 0 || col < 0 || row >= rows_ || col >= cols_)
			return false;

		buffer_[(row * cols_ + col) * offset_] = val;

		return true;
	}

	CUDAH bool getCol(int col, Matrix *output) {
		if (col >= cols_ || col < 0 || output->rows_ != rows_)
			return false;

		for (int i = 0; i < rows_; i++) {
			output->set(i, 0, buffer_[(i * cols_ + col) * offset_]);
		}

		return true;
	}

	CUDAH Matrix getCol(int col) {
		if (col < 0 || col >= cols_)
			return Matrix();

		return Matrix(rows_, 1, offset_ * cols_, buffer_ + col * offset_);
	}

	CUDAH bool getRow(int row, Matrix *output) {
		if (row > rows_ || row < 0 || output->cols_ != cols_)
			return false;

		for (int i = 0; i < cols_; i++) {
			output->set(0, i, buffer_[(row * cols_ + i) * offset_]);
		}

		return true;
	}

	CUDAH Matrix getRow(int row) {
		if (row < 0 || row >= rows_)
			return Matrix();

		return Matrix(1, cols_, offset_, buffer_ + row * cols_ * offset_);
	}

	/* Get sub matrix by removing row row and column col 
	 * from the current matrix.
	 */
	CUDAH bool getSubMatrix(int row, int col, Matrix *output) {
		if (row >= rows_ || row < 0 || col >= cols_ || col < 0)
			return false;

		//Get upper left quater
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				output->set(i, j, buffer_[(i * cols_ + j) * offset_]);
			}
		}

		//Get upper righ quater
		for (int i = 0; i < row; i++) {
			for (int j = col + 1; j < cols_; j++) {
				output->set(i, j - 1, buffer_[(i * cols_ + j) * offset_]);
			}
		}

		//Get lower left quater
		for (int i = row + 1; i < rows_; i++) {
			for (int j = 0; j < row; j++) {
				output->set(i - 1, j, buffer_[(i * cols_ + j) * offset_]);
			}
		}

		//Get lower right quater
		for (int i = row + 1; i < rows_; i++) {
			for (int j = col + 1; j < cols_; j++) {
				output->set(i - 1, j - 1, buffer_[(i * cols_ + j) * offset_]);
			}
		}

		return true;
	}

	static CUDAH bool add(const Matrix input0, const Matrix input1, Matrix& output) {
		if (input0.rows_ != input1.rows_ || input0.rows_ != output.rows_ || input0.cols_ != input1.cols_ || input0.cols_ != output.cols_)
			return false;

		for (int i = 0; i < input0.rows_; i++) {
			for (int j = 0; j < input0.cols_; j++) {
				output.set(i, j, input0.at(i, j) + input1.at(i, j));
			}
		}

		return true;
	}

	static CUDAH bool add(const Matrix input0, const Matrix input1, Matrix *output) {
		if (input0.rows_ != input1.rows_ || input0.rows_ != output->rows_ || input0.cols_ != input1.cols_ || input0.cols_ != output->cols_)
			return false;

		for (int i = 0; i < input0.rows_; i++) {
			for (int j = 0; j < input0.cols_; j++) {
				output->set(i, j, input0.at(i, j) + input1.at(i, j));
			}
		}

		return true;
	}

	static CUDAH bool subtract(const Matrix input0, const Matrix input1, Matrix& output) {
		if (input0.rows_ != input1.rows_ || input0.rows_ != output.rows_ || input0.cols_ != input1.cols_ || input0.cols_ != output.cols_)
			return false;

		for (int i = 0; i < input0.rows_; i++) {
			for (int j = 0; j < input0.cols_; j++) {
				output.set(i, j, input0.at(i, j) - input1.at(i, j));
			}
		}

		return true;
	}

	static CUDAH bool subtract(const Matrix input0, const Matrix input1, Matrix *output) {
		if (input0.rows_ != input1.rows_ || input0.rows_ != output->rows_ || input0.cols_ != input1.cols_ || input0.cols_ != output->cols_)
			return false;

		for (int i = 0; i < input0.rows_; i++) {
			for (int j = 0; j < input0.cols_; j++) {
				output->set(i, j, input0.at(i, j) - input1.at(i, j));
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

				output.set(i, j, tmp);
			}
		}

		return true;
	}

	static CUDAH bool multiply(const Matrix input0, const Matrix input1, Matrix *output) {
		if (input0.cols_ != input1.rows_ || input0.rows_ != output->rows_ || input1.cols_ != output->cols_)
			return false;

		for (int i = 0; i < output->rows_; i++) {
			for (int j = 0; j < output->cols_; j++) {
				float tmp = 0;
				for (int k = 0; k < input0.cols_; k++) {
					tmp += input0.at(i, k) * input1.at(k, j);
				}

				output->set(i, j, tmp);
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

	//Host-side methods
	bool operator!=(const Matrix mat) const;
	bool setValFromHost(int row, int col, float val);

protected:
	float *buffer_;
	int rows_, cols_, offset_;
};

}

#endif


