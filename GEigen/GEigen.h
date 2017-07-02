
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDAH __forceinline__ __host__ __device__

namespace GEigen {

class DiscreteMatrix {
public:
	CUDAH DiscreteMatrix() {
		ul_offset_ = ur_offset_ = ll_offset_ = lr_offset_ = 0;
		rows0_ = rows1_ = cols0_ = cols1_ = 0;
		upper_left_ = upper_right_ = lower_left_ = lower_right_ = NULL;
	}

	CUDAH DiscreteMatrix(int ul_offset, int ur_offset, int ll_offset, int lr_offset,
							int rows0, int rows1, int cols0, int cols1,
							float *upper_left, float *upper_right, float *lower_left, float *lower_right) {
		ul_offset_ = ul_offset;
		ur_offset_ = ur_offset;
		ll_offset_ = ll_offset;
		lr_offset_ = lr_offset;

		rows0_ = rows0;
		rows1_ = rows1;
		cols0_ = cols0;
		cols1_ = cols1;

		upper_left_ = upper_left;
		upper_right_ = upper_right;
		lower_left_ = lower_left;
		lower_right_ = lower_right;
	}

	CUDAH float at(int row, int col) {
		//Cell at upper left
		if (row < rows0_ && row >= 0 && col < cols0_ && col >= 0) {
			return upper_left_[(row * cols0_ + col) * ul_offset_];
		}

		//Cell at lower left
		if (row >= rows0_ && row < (rows0_ + rows1_) && col < cols0_ && col >= 0) {
			return lower_left_[((row - rows0_) * cols0_ + col) * ll_offset_];
		}

		//Cell at upper right
		if (row < rows0_ && row >= 0 && col >= cols0_ && col < (cols0_ + cols1_)) {
			return upper_right_[(row * cols0_ + col - cols0_) * ur_offset_];
		}

		//Cell at lower right
		if (row >= rows0_ && row < (rows0_ + rows1_) && col >= cols0_ && col < (cols0_ + cols1_)) {
			return lower_right_[((row - rows0_) * cols1_ + col - cols0_) * lr_offset_];
		}

		return -1;
	}

	CUDAH bool set(int row, int col, float val) {
		//Cell at upper left
		if (row < rows0_ && row >= 0 && col < cols0_ && col >= 0) {
			upper_left_[(row * cols0_ + col) * ul_offset_] = val;
			return true;
		}

		//Cell at lower left
		if (row >= rows0_ && row < (rows0_ + rows1_) && col < cols0_ && col >= 0) {
			lower_left_[((row - rows0_) * cols0_ + col) * ll_offset_] = val;
			return true;
		}

		//Cell at upper right
		if (row < rows0_ && row >= 0 && col >= cols0_ && col < (cols0_ + cols1_)) {
			upper_right_[(row * cols0_ + col - cols0_) * ur_offset_] = val;
			return true;
		}

		//Cell at lower right
		if (row >= rows0_ && row < (rows0_ + rows1_) && col >= cols0_ && col < (cols0_ + cols1_)) {
			lower_right_[((row - rows0_) * cols1_ + col - cols0_) * lr_offset_] = val;
			return true;
		}

		return false;
	}

	CUDAH DiscreteMatrix getCol(int col) {
		//Col at left half
		if (col < cols0_ && col >= 0) {
			return DiscreteMatrix(ul_offset_ * cols0_, 0, ll_offset_ * cols0_, 0, 
									rows0_, rows1_, 1, 0, 
									upper_left_ + col * ul_offset_, NULL, lower_left_ + col * ll_offset_, NULL);
		}


		//Col at right half
		if (col >= cols0_ && col < (cols0_ + cols1_)) {
			return DiscreteMatrix(ur_offset_ * cols1_, 0, lr_offset_ * cols1_, 0,
									rows0_, rows1_, 1, 0,
									upper_right_ + col * ur_offset_, NULL, lower_right_ + col * lr_offset_, NULL);
		}

		return DiscreteMatrix();
	}

	CUDAH DiscreteMatrix getRow(int row) {
		//Row at upper half
		if (row < rows0_ && row >= 0) {
			return DiscreteMatrix(ul_offset_, ur_offset_, 0, 0,
									1, 0, cols0_, cols1_, 
									upper_left_ + row * cols0_ * ul_offset_, upper_right_ + row * cols1_ * ur_offset_, NULL, NULL);
		}

		if (row > rows0_ && row < (rows0_ + rows1_)) {
			return DiscreteMatrix(ll_offset_, lr_offset_, 0, 0,
									1, 0, cols0_, cols1_,
									lower_left_ + (row - rows0_) * ll_offset_, lower_right_ + (row - rows0_) * lr_offset_, NULL, NULL);
		}

		return DiscreteMatrix();
	}

	CUDAH bool getSubMatrixFromUpperLeft(int row, int col, DiscreteMatrix *output) {
		if (row < 0 || col < 0)
			return false;

		//Get upper left of upper left sub matrix
		for (int i = 0; i < row && i < rows0_; i++) {
			for (int j = 0; j < col && j < cols0_; j++) {
				output->set(i, j, upper_left_[(i * cols0_ + j) * ul_offset_]);
			}
		}

		//Get upper right of upper left sub matrix
		for (int i = row + 1; i < rows0_; i++) {
			for (int j = 0; j < col && j < cols0_; j++) {
				output->set(i - 1, j, upper_left_[(i * cols0_ + j) * ul_offset_]);
			}
		}

		//Get lower right of upper left sub matrix
		for (int i = row + 1; i < rows0_; i++) {
			for (int j = col + 1; j < cols0_; j++) {
				output->set(i - 1, j - 1, upper_left_[(i * cols0_ + j) * ul_offset_]);
			}
		}

		//Get lower left of upper left sub matrix
		for (int i = 0; i < row && i < rows0_; i++) {
			for (int j = col + 1; j < cols0_; j++) {
				output->set(i, j - 1, upper_left_[(i * cols0_ + j) * ul_offset_]);
			}
		}

		return true;
	}

	CUDAH bool getSubMatrixFromLowerLeft(int row, int col, DiscreteMatrix *output) {
		if (row >= (rows0_ + rows1_) || col < 0)
			return false;

		for (int i = 0; i < row - rows0_; i++) {
			for (int j = 0; j < col && j < cols0_; j++) {
				output->set(i + rows0_, j, lower_left_[(i * cols0_ + j) * ll_offset_]);
			}
		}

		for (int i = 0; i < row - rows0_; i++) {
			for (int j = col + 1; j < cols0_; j++) {
				output->set(i + rows0_, j - 1, lower_left_[(i * cols0_ + j) * ll_offset_]);
			}
		}

		for (int i = ((row >= rows0_) ? row - rows0_ + 1 : 0); i < rows1_; i++) {
			for (int j = 0; j < col && j < cols0_; j++) {
				output->set(i + rows0_ - 1, j, lower_left_[(i * cols0_ + j) * ll_offset_]);
			}
		}

		for (int i = ((row >= rows0_) ? row - rows0_ + 1 : 0); i < rows1_; i++) {
			for (int j = col + 1; j < cols0_; j++) {
				output->set(i + rows0_ - 1, j - 1, lower_left_[(i * cols0_ + j) * ll_offset_]);
			}
		}

		return true;
	}

	CUDAH bool getSubMatrixFromUpperRight(int row, int col, DiscreteMatrix *output) {
		if (row < 0 || col >= (cols0_ + cols1_))
			return false;

		for (int i = 0; i < row && i < rows0_; i++) {
			for (int j = 0; j < col - cols0_; j++) {
				output->set(i, j + cols0_, upper_right_[(i * cols1_ + j) * ur_offset_]);
			}
		}

		for (int i = 0; i < row && i < rows0_; i++) {
			for (int j = ((col >= cols0_) ? col - cols0_ + 1 : 0); j < cols1_; j++) {
				output->set(i, j + cols0_ - 1, upper_right_[(i * cols1_ + j) * ur_offset_]);
			}
		}

		for (int i = row + 1; i < rows0_; i++) {
			for (int j = 0; j < col - cols0_; j++) {
				output->set(i, j + cols0_, upper_right_[(i * cols1_ + j) * ur_offset_]);
			}
		}

		for (int i = row + 1; i < rows0_; i++) {
			for (int j = ((col >= cols0_) ? col - cols0_ + 1 : 0); j < cols1_; j++) {
				output->set(i, j + cols0_ - 1, upper_right_[(i * cols1_ + j) * ur_offset_]);
			}
		}

		return true;
	}

	CUDAH bool getSubMatrixFromLowerRight(int row, int col, DiscreteMatrix *output) {
		if (row > (rows0_ + rows1_) || col >(cols0_ + cols1_))
			return false;

		for (int i = 0; i < row - rows0_; i++) {
			for (int j = 0; j < col - cols0_; j++) {
				output->set(i + rows0_, j + cols0_, lower_right_[(i * cols1_ + j) * lr_offset_]);
			}
		}

		for (int i = 0; i < row - rows0_; i++) {
			for (int j = ((col >= cols0_) ? col - cols0_ + 1 : 0); j < cols1_; j++) {
				output->set(i + rows0_, j + cols0_ - 1, lower_right_[(i * cols1_ + j) * lr_offset_]);
			}
		}

		for (int i = ((row >= rows0_) ? row - rows0_ + 1 : 0); i < rows1_; i++) {
			for (int j = 0; j < col - cols0_; j++) {
				output->set(i + rows0_, j + cols0_ - 1, lower_right_[(i * cols1_ + j) * lr_offset_]);
			}
		}

		for (int i = ((row >= rows0_) ? row - rows0_ + 1 : 0); i < rows1_; i++) {
			for (int j = ((col >= cols0_) ? col - cols0_ + 1 : 0); j < cols1_; j++) {
				output->set(i + rows0_ - 1, j + cols0_ - 1, lower_right_[(i * cols1_ + j) * lr_offset_]);
			}
		}

		return true;
	}

	CUDAH bool getSubMatrix(int row, int col, DiscreteMatrix *output) {
		if (!getSubMatrixFromUpperLeft(row, col, output))
			return false;

		if (!getSubMatrixFromUpperRight(row, col, output))
			return false;

		if (!getSubMatrixFromLowerLeft(row, col, output))
			return false;

		if (!getSubMatrixFromLowerRight(row, col, output))
			return false;

		return true;
	}

	CUDAH float det() {

	}

private:
	float *upper_left_, *upper_right_, *lower_left_, *lower_right_;
	int ul_offset_, ur_offset_, ll_offset_, lr_offset_;
	int rows0_, rows1_, cols0_, cols1_;
};

class Matrix {
public:
	Matrix();
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

	CUDAH float at(int row, int col) const {
		return buffer_[(row * cols_ + col) * offset_];
	}

	CUDAH void set(int row, int col, float val) {
		buffer_[(row * cols_ + col) * offset_] = val;
	}

	CUDAH bool getCol(int col, Matrix *output) {
		if (col > cols_ || col < 0 || output->rows_ != rows_)
			return false;

		for (int i = 0; i < rows_; i++) {
			output->set(i, 0, buffer_[(i * cols_ + col) * offset_]);
		}

		return true;
	}

	CUDAH Matrix getCol(int col) {
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

	CUDAH float det() {

	}

private:
	float *buffer_;
	int rows_, cols_, offset_;
}



}


