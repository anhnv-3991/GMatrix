
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDAH __forceinline__ __host__ __device__

namespace GEigen {

class Matrix {
public:
	CUDAH Matrix() {
		buffer_ = NULL;
		rows_ = cols_ = offset_ = 0;
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
protected:
	float *buffer_;
	int rows_, cols_, offset_;
};

class SquareMatrix : protected Matrix {
public:
	CUDAH SquareMatrix() {
		rows_ = cols_ = 0;
		offset_ = 0;
		buffer_ = NULL;
	}

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

		for (int i = 0, float sign = 1; i < rows_; i++, sign *= (-1)) {
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

		for (int i = 0, float sign = 1; i < rows_; i++) {
			for (int j = 0; j < cols_; j++, sign *= (-1)) {
				SquareMatrix sub_matrix(rows_ - 1, offset_, temp_buffer);

				getSubMatrix(i, j, &sub_matrix);

				output->set(i, j, sign * 1 / det_val * sub_matrix.det(temp_buffer + (rows_ - 1) * (cols_ - 1) * offset_));
			}
		}

		return true;
	}
};

class IdentityMatrix : protected SquareMatrix {
public:
	CUDAH IdentityMatrix() {
		rows_ = cols_ = 0;
		offset_ = 0;
		buffer_ = NULL;
	}

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
	}
};


}


