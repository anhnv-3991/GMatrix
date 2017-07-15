#include "Registration.h"
#include "debug.h"

namespace gpu {

GRegistration::GRegistration()
{
	max_iterations_ = 0;
	x_ = y_ = z_ = NULL;
	points_number_ = 0;
		
	trans_x_ = trans_y_ = trans_z_ = NULL;

	converged_ = false;
	nr_iterations_ = 0;

	transformation_epsilon_ = 0;
	target_cloud_updated_ = true;
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

void GRegistration::setInputTarget(float *target_x, float *target_y, float *target_z, int points_number)
{
	target_points_number_ = points_number;

	if (target_points_number_ != 0) {
		checkCudaErrors(cudaMalloc(&target_x_, sizeof(float) * target_points_number_));
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));
		checkCudaErrors(cudaMalloc(&target_y_, sizeof(float) * target_points_number_));

		checkCudaErrors(cudaMemcpy(target_x_, target_x, sizeof(float) * target_points_number_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(target_y_, target_y, sizeof(float) * target_points_number_, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(target_z_, target_z, sizeof(float) * target_points_number_, cudaMemcpyHostToDevice));
	}
}


void GRegistration::freeInputTarget()
{
	if (target_x_ != NULL)
		checkCudaErrors(cudaFree(target_x_));

	if (target_y_ != NULL)
		checkCudaErrors(cudaFree(target_y_));

	if (target_z_ != NULL)
		checkCudaErrors(cudaFree(target_z_));

	target_points_number_ = 0;
}

void GRegistration::align(Eigen::Matrix<float, 4, 4> &guess)
{
	converged_ = false;

	final_transformation_ = transformation_ = previous_transformation_ = Eigen::Matrix<float, 4, 4>::Identity();

	computeTransformation(guess);
}

}


