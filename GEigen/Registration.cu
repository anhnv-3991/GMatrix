#include "Registration.h"
#include "debug.h"

namespace gpu {

GRegistration::GRegistration()
{
	max_iterations_ = 0;
	x_ = y_ = z_ = NULL;
	points_number_ = 0;
		
	trans_x_ = trans_y_ = trans_z_ = NULL;
	out_points_num_ = 0;

	converged_ = false;
	nr_iterations_ = 0;

	transformation_epsilon_ = 0;
	target_cloud_updated_ = true;
}

void GRegistration::setTransformationEpsilon(double trans_eps)
{
	transformation_epsilon_ = trans_eps;
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

		checkCudaErrors(cudaMalloc(&trans_x_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&trans_y_, sizeof(float) * points_number_));
		checkCudaErrors(cudaMalloc(&trans_z_, sizeof(float) * points_number_));
	}
}


void GRegistration::align(Eigen::Matrix<float, 4, 4> &guess)
{
	if (!initCompute())
		return;

	converged_ = false;

	final_transformation_ = transformation_ = previous_transformation_ = Eigen::Matrix<float, 4, 4>::Identity();

	computeTransformation(guess);

	//deinitCompute()
}

bool GRegistration::initCompute()
{
	if (points_number_ == 0 || x_ == NULL || y_ == NULL || z_ == NULL) {
		fprintf(stderr, "No input target dataset was given!\n");
		return false;
	}

	if (target_cloud_updated_ && !force_no_recompute_) {
		voxel_grid_.setInput(x_, y_, z_, points_number_);
		target_cloud_updated_ = false;
	}

	return true;
}
}


