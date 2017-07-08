#include "NormalDistributionTransform.h"
#include "debug.h"
#include <cmath>

namespace gpu {



void GNormalDistributionTransform::computeTransformation()
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

	point_gradient_ = MatrixHost(3, 6);
	point_hessian_ = MatrixHost(18, 6);

	point_gradient_(0, 0) = 1;
	point_gradient_(1, 1) = 1;
	point_gradient_(2, 2) = 1;

	MatrixDevice p(6, 1), delta_p(6, 1), score_gradient(6, 1);
	MatrixHost p_host(6, 1), delta_p_host(6, 1), score_gradient_host(6, 1);

	Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;

	Eigen::Matrix<float, 4, 4> eig_transformation_tmp;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			eig_transformation_tmp(i, j) = host_final_transformation_(i, j);
		}
	}

	eig_transformation.matrix() = eig_transformation_tmp;

	Eigen::Matrix<double, 6, 1> p_tmp, delta_p_tmp, score_gradient_tmp;
	Eigen::Vector3f init_trans_tmp = eig_transformation.translation();
	Eigen::Vector3f init_rot_tmp = eig_transformation.rotation().eulerAngles(0, 1, 2);
	p_tmp << init_trans_tmp(0), init_trans_tmp(1), init_trans_tmp(2), init_rot_tmp(0), init_rot_tmp(1), init_rot_tmp(2);

	for (int i = 0; i < p.rows(); i++) {
		for (int j = 0; j < p.cols(); j++) {
			p_host(i, j) = p_tmp(i, j);
		}
	}
}

double GNormalDistributionTransform::computeDerivatives(MatrixDevice score_gradient, MatrixDevice hessian,
														float *source_x, float *source_y, float *source_z,
														MatrixDevice pose, bool compute_hessian)
{

}

void GNormalDistributionTransform::computeAngleDerivatives(MatrixHost pose, bool compute_hessian)
{
	double cx, cy, cz, sx, sy, sz;

	if (fabs(pose(3)) < 10e-5) {
		cx = 1.0;
		sx = 0.0;
	} else {
		cx = cos(pose(3));
		sx = sin(pose(3));
	}

	if (fabs(pose(4)) < 10e-5) {
		cy = 1.0;
		sy = 0.0;
	} else {
		cy = cos(pose(4));
		sy = sin(pose(4));
	}

	if (fabs(pose(5)) < 10e-5) {
		cz = cos(pose(5));
		sz = sin(pose(5));
	}

	j_ang_a_ = MatrixHost(3, 1);
	j_ang_b_ = MatrixHost(3, 1);
	j_ang_c_ = MatrixHost(3, 1);
	j_ang_d_ = MatrixHost(3, 1);
	j_ang_e_ = MatrixHost(3, 1);
	j_ang_f_ = MatrixHost(3, 1);
	j_ang_g_ = MatrixHost(3, 1);
	j_ang_h_ = MatrixHost(3, 1);

	j_ang_a_(0) = -sx * sz + cx * sy * cz;
	j_ang_a_(1) = -sx * cz - cx * sy * sz;
	j_ang_a_(2) = -cx * cy;

	j_ang_b_(0) = cx * sz + sx * sy * cz;
	j_ang_b_(1) = cx * cz - sx * sy * sz;
	j_ang_b_(2) = -sx * cy;

	j_ang_c_(0) = -sy * cz;
	j_ang_c_(1) = sy * sz;
	j_ang_c_(2) = cy;

	j_ang_d_(0) = sx * cy * cz;
	j_ang_d_(1) = -sx * cy * sz;
	j_ang_d_(2) = sx * sy;

	j_ang_e_(0) = -cx * cy * cz;
	j_ang_e_(1) = cx * cy * sz;
	j_ang_e_(2) = -cx * sy;

	j_ang_f_(0) = -cy * sz;
	j_ang_f_(1) = -cy * cz;
	j_ang_f_(2) = 0;

	j_ang_g_(0) = cx * cz - sx * sy * sz;
	j_ang_g_(1) = -cx * sz - sx * sy * cz;
	j_ang_g_(2) = 0;

	j_ang_h_(0) = sx * cz + cx * sy * sz;
	j_ang_h_(1) = cx * sy * cz - sx * sz;
	j_ang_h_(2) = 0;

	if (compute_hessian) {
		h_ang_a2_(0) = -cx * sz - sx * sy * cz;
		h_ang_a2_(1) = -cx * cz + sx * sy * sz;
		h_ang_a3_(2) = sx * cy;

		h_ang_a3_(0) = -sx * sz + cx * sy * cz;
		h_ang_a3_(1) = -cx * sy * sz - sx * cz;
		h_ang_a3_(2) = -cx * cy;

		h_ang_b2_(0) = cx * cy * cz;
		h_ang_b2_(1) = -cx * cy * sz;
		h_ang_b2_(2) = cx * sy;

		h_ang_b3_(0) = sx * cy * cz;
		h_ang_b3_(1) = -sx * cy * sz;
		h_ang_b3_(2) = sx * sy;

		h_ang_c2_(0) = -sx * cz - cx * sy * sz;
		h_ang_c2_(1) = sx * sz - cx * sy * cz;
		h_ang_c2_(2) = 0;

		h_ang_c3_(0) = cx * cz - sx * sy * sz;
		h_ang_c3_(1) = -sx * sy * cz - cx * sz;
		h_ang_c3_(2) = 0;

		h_ang_d1_(0) = -cy * cz;
		h_ang_d1_(1) = cy * sz;
		h_ang_d1_(2) = sy;

		h_ang_d2_(0) = -sx * sy * cz;
		h_ang_d2_(1) = sx * sy * sz;
		h_ang_d2_(2) = sx * cy;

		h_ang_d3_(0) = cx * sy * cz;
		h_ang_d3_(1) = -cx * sy * sz;
		h_ang_d3_(2) = -cx * cy;

		h_ang_e1_(0) = sy * sz;
		h_ang_e1_(1) = sy * cz;
		h_ang_e1_(3) = 0;

		h_ang_e2_(0) = -sx * cy * sz;
		h_ang_e2_(1) = -sx * cy * cz;
		h_ang_e2_(2) = 0;

		h_ang_e3_(0) = cx * cy * sz;
		h_ang_e3_(1) = cx * cy * cz;
		h_ang_e3_(2) = 0;

		h_ang_f1_(0) = -cy * cz;
		h_ang_f1_(1) = cy * sz;
		h_ang_f1_(2) = 0;

		h_ang_f2_(0) = -cx * sz - sx * sy * cz;
		h_ang_f2_(1) = -cx * cz + sx * sy * sz;
		h_ang_f2_(2) = 0;

		h_ang_f3_(0) = -sx * sz + cx * sy * cz;
		h_ang_f3_(1) = -cx * sy * sz - sx * cz;
		h_ang_f3_(2) = 0;
	}

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
