#include "Registration.h"
#include "debug.h"

namespace gpu {

GRegistration::GRegistration()
{
	trans_epsilon_ = 0;
	step_size_ = 0;
	resolution_ = 0;
	max_iterations_ = 0;
	x_ = y_ = z_ = NULL;
	points_number_ = 0;
		
	out_x_ = out_y_ = out_z_ = NULL;
	out_points_num_ = 0;

	converged_ = false;
	nr_iterations_ = 0;
}

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
	float const *ibuffer = input.getBuffer();
	float *obuffer = NULL;

	if (rows > 0 && cols > 0) {
		checkCudaErrors(cudaMalloc(&obuffer, sizeof(float) * rows * cols * offset));
		checkCudaErrors(cudaMemcpy(obuffer, ibuffer, sizeof(float) * rows * cols * offset, cudaMemcpyHostToDevice));

		init_guess_ = Matrix(rows, cols, offset, obuffer);
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


