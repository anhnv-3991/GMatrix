#include "NormalDistributionTransform.h"
#include "debug.h"
#include <cmath>

namespace gpu {



void GNormalDistributionTransform::computeTransform()
{
	nr_iterations_ = 0;
	converged_ = false;

	double gauss_c1, gauss_c2, gauss_d3;

	gauss_c1 = 10 * ( 1 - outlier_ratio_);
	gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
	gauss_d3 = - log(gauss_c2);
	gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3;
	gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1_);

	if (init_guess_ != IdentityMatrix(4)) {
		final_transformation_ = init_guess_;
		
		transformPointCloud(out_x_, out_y_, out_z_, out_x_, out_y_, out_z_, points_number_, init_guess_);
	}

	point_gradient_ = Matrix(3, 6);
	point_hessian_ = Matrix(18, 6);

	point_gradient_.setValFromHost(0, 0, 1);
	point_gradient_.setValFromHost(1, 1, 1);
	point_gradient_.setValFromHost(2, 2, 1);

	Matrix p(6, 1), delta_p(6, 1), score_gradient(6, 1);
}

extern "C" __global__ void transformPointCloud(float *in_x, float *in_y, float *in_z,
												float *out_x, float *out_y, float *out_z,
												int point_num, Matrix transform)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float x, y, z;

	for (int i = idx; i < point_num; i += stride) {
		x = in_x[i];
		y = in_y[i];
		z = in_z[i];
		out_x[i] = transform(0, 0) * x + transform(0, 1) * y + transform(0, 2) * z;
		out_y[i] = transform(1, 0) * x + transform(1, 1) * y + transform(1, 2) * z;
		out_z[i] = transform(2, 0) * x + transform(2, 1) * y + transform(2, 2) * z;
	}
}

void GNormalDistributionTransform::transformPointCloud(float *in_x, float *in_y, float *in_z,
														float *out_x, float *out_y, float *out_z,
														int points_number, const Matrix transform)
{
	if (points_number > 0) {
		int block_x = (points_number <= BLOCK_SIZE_X) ? points_number : BLOCK_SIZE_X;
		int grid_x = (points_number - 1) / block_x + 1;

		transformPointCloud <<<grid_x, block_x >>>(in_x, in_y, in_z, out_x, out_y, out_z, points_number, transform);
		checkCudaErrors(cudaDeviceSynchronize());
	}
}
}
