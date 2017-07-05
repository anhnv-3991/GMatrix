#include "Registration.h"
#include "debug.h"

namespace gpu {

void GRegistration::setTransformationEpsilon(double trans_eps)
{
	trans_epsilon_ = trans_eps;
}

void GRegistration::setStepSize(double step_size)
{
	step_size_ = step_size;
}

void GRegistration::setResolution(float resolution)
{
	resolution_ = resolution;
}

void GRegistration::setMaximumIterations(int max_itr)
{
	max_iterations_ = max_itr;
}

void GRegistration::setInputSource(float *x, float *y, float *z, int points_num)
{
	if (points_num > 0) {
		points_number_ = points_num;

		checkCudaErrors(cudaMalloc(&x_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&y_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&z_, sizeof(float) * points_number_));

		checkCudaErrors(cudaMemcpy(x_, x, sizeof(float) * points_number_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(y_, y, sizeof(float) * points_number_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(z_, z, sizeof(float) * points_number_, cudaMemcpyHostToDevice));
	}
}

void GRegistration::setInitGuess(const Matrix input)
{
	int rows = input.getRowsCount();
	int cols = input.getColsCount();
	int offset = input.getOffset();
	float *ibuffer = input.getBuffer();
	float *obuffer = NULL;

	if (row > 0 && col > 0) {
		checkCudaErrors(cudaMalloc(&obuffer, sizeof(float) * rows * cols * offset));
		checkCudaErrors(cudaMemcpy(obuffer, ibuffer, sizeof(float) * rows * cols * offset, cudaMemcpyHostToDevice));

		init_guess = Matrix(rows, cols, offset, obuffer);
	}
}

void GRegistration::align()
{
	//initCompute() ???

	converged_ = false;

	final_transformation_ = IdentityMatrix(4);
	transformation_ = IdentityMatrix(4);
	previous_transformation_ = IdentityMatrix(4);

	computeTransformation();

	//deinitCompute()
}

}


