#include "NormalDistributionsTransform.h"
#include "debug.h"
#include <cmath>

namespace gpu {
void GNormalDistributionsTransform::setInputTarget(float *target_x, float *target_y, float *target_z, int points_number)
{
	GRegistration::setInputTarget(target_x, target_y, target_z, points_number);

	if (points_number != 0)
		voxel_grid_.setInput(target_x_, target_y_, target_z_, target_points_number_);
}

void GNormalDistributionsTransform::computeTransformation(Eigen::Matrix<float, 4, 4> &guess)
{
	nr_iterations_ = 0;
	converged_ = false;

	double gauss_c1, gauss_c2, gauss_d3;

	gauss_c1 = 10 * ( 1 - outlier_ratio_);
	gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
	gauss_d3 = - log(gauss_c2);
	gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3;
	gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1_);

	if (guess != Eigen::Matrix4f::Identity()) {
		final_transformation_ = guess;
		
		transformPointCloud(x_, y_, z_, trans_x_, trans_y_, trans_z_, points_number_, guess);
	}

	Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
	eig_transformation.matrix() = final_transformation_;

	Eigen::Matrix<double, 6, 1> p, delta_p, score_gradient;
	Eigen::Vector3f init_translation = eig_transformation.translation();
	Eigen::Vector3f init_rotation = eig_transformation.rotation().eulerAngles(0, 1, 2);

	p << init_translation(0), init_translation(1), init_translation(2), init_rotation(0), init_rotation(1), init_rotation(2);

	Eigen::Matrix<double, 6, 6> hessian;

	double score = 0;
	double delta_p_norm;

	score = computeDerivatives(score_gradient, hessian, trans_x_, trans_y_, trans_z_, points_number_, p);

	while (!converged_) {
		previous_transformation_ = transformation_;

		Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);

		delta_p = sv.solve(-score_gradient);

		delta_p_norm = delta_p.norm();

		if (delta_p_norm == 0 || delta_p_norm != delta_p_norm) {
			trans_probability_ = score / static_cast<double>(points_number_);
			converged_ = delta_p_norm == delta_p_norm;
			return;
		}

		delta_p.normalize();
		delta_p_norm = computeStepLengthMT(p, delta_p, delta_p_norm, step_size_, transformation_epsilon_ / 2, score, score_gradient, hessian, trans_x_, trans_y_, trans_z_, points_number_);
		delta_p *= delta_p_norm;

		transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(delta_p(0)), static_cast<float>(delta_p(1)), static_cast<float>(delta_p(2))) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(3)), Eigen::Vector3f::UnitX()) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(4)), Eigen::Vector3f::UnitY()) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(5)), Eigen::Vector3f::UnitZ())).matrix();

		p = p + delta_p;

		//Not update visualizer

		if (nr_iterations_ > max_iterations_ || (nr_iterations_ && (std::fabs(delta_p_norm) < transformation_epsilon_)))
			converged_ = true;

		nr_iterations_++;
	}

	trans_probability_ = score / static_cast<double>(points_number_);
}

extern "C" __global__ void matrixListInit(MatrixDevice *matrix, double *matrix_buff, int matrix_num, int rows, int cols)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < matrix_num; i += stride) {
		matrix[i].setRows(rows);
		matrix[i].setCols(cols);
		matrix[i].setOffset(matrix_num);
		matrix[i].setBuffer(matrix_buff + i);

		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				matrix[i](j, k) = 0;
			}
		}
	}
}

extern "C" __global__ void computePointDerivatives(float *x, float *y, float *z, int points_num,
													int *valid_points, int valid_points_num,
													MatrixDevice j_ang_a, MatrixDevice j_ang_b, MatrixDevice j_ang_c, MatrixDevice j_ang_d,
													MatrixDevice j_ang_e, MatrixDevice j_ang_f, MatrixDevice j_ang_g, MatrixDevice j_ang_h,
													MatrixDevice h_ang_a2, MatrixDevice h_ang_a3, MatrixDevice h_ang_b2, MatrixDevice h_ang_b3, MatrixDevice h_ang_c2,
													MatrixDevice h_ang_c3, MatrixDevice h_ang_d1, MatrixDevice h_ang_d2, MatrixDevice h_ang_d3, MatrixDevice h_ang_e1,
													MatrixDevice h_ang_e2, MatrixDevice h_ang_e3, MatrixDevice h_ang_f1, MatrixDevice h_ang_f2, MatrixDevice h_ang_f3,
													MatrixDevice *point_gradients, MatrixDevice *point_hessians, bool compute_hessian)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = valid_points[i];

		//Orignal coordinates
		float o_x = x[pid];
		float o_y = y[pid];
		float o_z = z[pid];

		MatrixDevice pg = point_gradients[i];		//3x6 Matrix

		//Compute point derivatives
		pg(1, 3) = o_x * j_ang_a(0) + o_y * j_ang_a(1) + o_z * j_ang_a(2);
		pg(2, 3) = o_x * j_ang_b(0) + o_y * j_ang_b(1) + o_z * j_ang_b(2);
		pg(0, 4) = o_x * j_ang_c(0) + o_y * j_ang_c(1) + o_z * j_ang_c(2);
		pg(1, 4) = o_x * j_ang_d(0) + o_y * j_ang_d(1) + o_z * j_ang_d(2);
		pg(2, 4) = o_x * j_ang_e(0) + o_y * j_ang_e(1) + o_z * j_ang_e(2);
		pg(0, 5) = o_x * j_ang_f(0) + o_y * j_ang_f(1) + o_z * j_ang_f(2);
		pg(1, 5) = o_x * j_ang_g(0) + o_y * j_ang_g(1) + o_z * j_ang_g(2);
		pg(2, 5) = o_x * j_ang_h(0) + o_y * j_ang_h(1) + o_z * j_ang_h(2);

		if (compute_hessian) {
			MatrixDevice ph = point_hessians[i];		//18x6 Matrix

			ph(9, 3) = 0;
			ph(10, 3) = o_x * h_ang_a2(0) + o_y * h_ang_a2(1) + o_z * h_ang_a2(2);
			ph(11, 3) = o_x * h_ang_a3(0) + o_y * h_ang_a3(1) + o_z * h_ang_a3(2);

			ph(12, 3) = ph(9, 4) = 0;
			ph(13, 3) = ph(10, 4) = o_x * h_ang_b2(0) + o_y * h_ang_b2(1) + o_z * h_ang_b2(2);
			ph(14, 3) = ph(11, 4) = o_x * h_ang_b3(0) + o_y * h_ang_b3(1) + o_z * h_ang_b3(2);

			ph(15, 3) = 0;
			ph(16, 3) = ph(9, 5) = o_x * h_ang_c2(0) + o_y * h_ang_c2(1) + o_z * h_ang_c2(2);
			ph(17, 3) = ph(10, 5) = o_x * h_ang_c3(0) + o_y * h_ang_c3(1) + o_z * h_ang_c3(2);

			ph(12, 4) = o_x * h_ang_d1(0) + o_y * h_ang_d1(1) + o_z * h_ang_d1(2);
			ph(13, 4) = o_x * h_ang_d2(0) + o_y * h_ang_d2(1) + o_z * h_ang_d2(2);
			ph(14, 4) = o_x * h_ang_d3(0) + o_y * h_ang_d3(1) + o_z * h_ang_d3(2);

			ph(15, 4) = ph(12, 5) = o_x * h_ang_e1(0) + o_y * h_ang_e1(1) + o_z * h_ang_e1(2);
			ph(16, 4) = ph(13, 5) = o_x * h_ang_e2(0) + o_y * h_ang_e2(1) + o_z * h_ang_e2(2);
			ph(17, 4) = ph(14, 5) = o_x * h_ang_e3(0) + o_y * h_ang_e3(1) + o_z * h_ang_e3(2);

			ph(15, 5) = o_x * h_ang_f1(0) + o_y * h_ang_f1(1) + o_z * h_ang_f1(2);
			ph(16, 5) = o_x * h_ang_f2(0) + o_y * h_ang_f2(1) + o_z * h_ang_f2(2);
			ph(17, 5) = o_x * h_ang_f3(0) + o_y * h_ang_f3(1) + o_z * h_ang_f3(2);
		}
	}
}

extern "C" __global__ void computeDerivative(float *trans_x, float *trans_y, float *trans_z, int points_num,
												int *valid_points, int *voxel_id, int valid_points_num,
												GVoxel *grid, double gauss_d1, double gauss_d2,
												MatrixDevice *point_gradients, MatrixDevice *point_hessians,
												MatrixDevice *score_gradients, MatrixDevice *hessians,
												double *score, bool compute_hessian)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = valid_points[i];

		//Transformed coordinates
		float t_x = trans_x[pid];
		float t_y = trans_y[pid];
		float t_z = trans_z[pid];

		MatrixDevice pg = point_gradients[i];		//3x6 Matrix
		MatrixDevice sg = score_gradients[i];		//6x1 Matrix

		double score_inc = 0;

		for ( int vid = voxel_id[i]; vid < voxel_id[i + 1]; vid++) {
			GVoxel *voxel = grid + vid;
			MatrixDevice centroid = voxel->centroid();
			MatrixDevice icov = voxel->inverseCovariance();	//3x3 matrix

			double cov_dxd_pi_x, cov_dxd_pi_y, cov_dxd_pi_z;

			t_x -= centroid(0);
			t_y -= centroid(1);
			t_z -= centroid(2);

			double e_x_cov_x = expf(-gauss_d2 * ((t_x * icov(0, 0) + t_y * icov(1, 0) + t_z * icov(2, 0)) * t_x
												+ ((t_x * icov(0, 1) + t_y * icov(1, 1) + t_z * icov(2, 1)) * t_y)
												+ ((t_x * icov(0, 2) + t_y * icov(1, 2) + t_z * icov(2, 2)) * t_z)) / 2);
			score_inc += -gauss_d1 * e_x_cov_x;

			e_x_cov_x *= gauss_d2;

			e_x_cov_x *= gauss_d1;

			for (int n = 0; n < 6; n++) {
				cov_dxd_pi_x = icov(0, 0) * pg(0, n) + icov(0, 1) * pg(1, n) + icov(0, 2) * pg(2, n);
				cov_dxd_pi_y = icov(1, 0) * pg(0, n) + icov(1, 1) * pg(1, n) + icov(1, 2) * pg(2, n);
				cov_dxd_pi_z = icov(2, 0) * pg(0, n) + icov(2, 1) * pg(1, n) + icov(2, 2) * pg(2, n);

				sg(n) += (t_x * cov_dxd_pi_x + t_y * cov_dxd_pi_y + t_z * cov_dxd_pi_z) * e_x_cov_x;

				//Compute hessian
				if (compute_hessian) {
					MatrixDevice ph = point_hessians[i];		//18x6 Matrix
					MatrixDevice h = hessians[i];				//6x6 Matrix

					for (int p = 0; p < h.cols(); p++) {
						h(n, p) += e_x_cov_x * (-gauss_d2 * (t_x * cov_dxd_pi_x + t_y * cov_dxd_pi_y + t_z * cov_dxd_pi_z) *
													(t_x * (icov(0, 0) * pg(0, p) + icov(0, 1) * pg(1, p) + icov(0, 2) * pg(2, p))
													+ t_y * (icov(1, 0) * pg(0, p) + icov(1, 1) * pg(1, p) + icov(1, 2) * pg(2, p))
													+ t_z * (icov(2, 0) * pg(0, p) + icov(2, 1) * pg(1, p) + icov(2, 2) * pg(2, p)))
													+ (t_x * (icov(0, 0) * ph(3 * n, p) + icov(0, 1) * ph(3 * n + 1, p) + icov(0, 2) * ph(3 * n + 2, p))
													+ t_y * (icov(1, 0) * ph(3 * n, p) + icov(1, 1) * ph(3 * n + 1, p) + icov(1, 2) * ph(3 * n + 2, p))
													+ t_z * (icov(2, 0) * ph(3 * n, p) + icov(2, 1) * ph(3 * n + 1, p) + icov(2, 2) * ph(3 * n + 2, p)))
													+ (pg(0, p) * cov_dxd_pi_x + pg(1, p) * cov_dxd_pi_y + pg(2, p) * cov_dxd_pi_z));
					}
				}
			}
		}

		score[i] = score_inc;
	}
}

/* Compute sum of a list of a matrixes */
extern "C" __global__ void matrixSum(MatrixDevice *matrix_list, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride) {
		MatrixDevice left = matrix_list[i];
		MatrixDevice right = (i + half_size < full_size) ? matrix_list[i + half_size] : MatrixDevice();

		if (!right.isEmpty()) {
			for (int j = 0; j < left.rows(); j++) {
				for (int k = 0; k < left.cols(); k++) {
					left(j, k) += right(j, k);
				}
			}
		}
	}
}

extern "C" __global__ void sumScore(double *score, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride) {
		score[i] += (i + half_size < full_size) ? score[i + half_size] : 0;
	}
}

double GNormalDistributionsTransform::computeDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient, Eigen::Matrix<double, 6, 6> &hessian,
														float *trans_x, float *trans_y, float *trans_z,
														int points_num, Eigen::Matrix<double, 6, 1> pose, bool compute_hessian)
{
	MatrixHost p(6, 1);

	for (int i = 0; i < 6; i++) {
		p(i) = pose(i, 0);
	}
	//Compute Angle Derivatives
	computeAngleDerivatives(p, compute_hessian);
	//Radius Search
	voxel_grid_.radiusSearch(trans_x, trans_y, trans_z, points_num, resolution_, INT_MAX);

	int *valid_points = voxel_grid_.getValidPoints();

	int *voxel_id = voxel_grid_.getVoxelIds();

	int search_size = voxel_grid_.getSearchResultSize();

	int valid_points_num = voxel_grid_.getValidPointsNum();

	GVoxel *voxel_list = voxel_grid_.getVoxelList();

	int voxel_num = voxel_grid_.getVoxelNum();

	float max_x = voxel_grid_.getMaxX();
	float max_y = voxel_grid_.getMaxY();
	float max_z = voxel_grid_.getMaxZ();

	float min_x = voxel_grid_.getMinX();
	float min_y = voxel_grid_.getMinY();
	float min_z = voxel_grid_.getMinZ();

	float voxel_x = voxel_grid_.getVoxelX();
	float voxel_y = voxel_grid_.getVoxelY();
	float voxel_z = voxel_grid_.getVoxelZ();

	int max_b_x = voxel_grid_.getMaxBX();
	int max_b_y = voxel_grid_.getMaxBY();
	int max_b_z = voxel_grid_.getMaxBZ();

	int min_b_x = voxel_grid_.getMinBX();
	int min_b_y = voxel_grid_.getMinBY();
	int min_b_z = voxel_grid_.getMinBZ();

	//Update score gradient and hessian matrix
	MatrixDevice *gradients_list, *hessians_list, *points_gradient, *points_hessian;

	checkCudaErrors(cudaMalloc(&gradients_list, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&hessians_list, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&points_gradient, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&points_hessian, sizeof(MatrixDevice) * valid_points_num));

	double *gradient_buff, *hessian_buff, *points_gradient_buff, *points_hessian_buff, *score;

	checkCudaErrors(cudaMalloc(&gradient_buff, sizeof(double) * valid_points_num * 6));
	checkCudaErrors(cudaMalloc(&hessian_buff, sizeof(double) * valid_points_num * 6 * 6));
	checkCudaErrors(cudaMalloc(&points_gradient_buff, sizeof(double) * valid_points_num * 3 * 6));
	checkCudaErrors(cudaMalloc(&points_hessian_buff, sizeof(double) * valid_points_num * 18 * 6));
	checkCudaErrors(cudaMalloc(&score, sizeof(double) * valid_points_num));

	int block_x = (valid_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : valid_points_num;
	int grid_x = (valid_points_num - 1) / block_x + 1;

	matrixListInit<<<grid_x, block_x>>>(gradients_list, gradient_buff, valid_points_num, 1, 6);
	matrixListInit<<<grid_x, block_x>>>(hessians_list, hessian_buff, valid_points_num, 6, 6);
	matrixListInit<<<grid_x, block_x>>>(points_gradient, points_gradient_buff, valid_points_num, 3, 6);
	matrixListInit<<<grid_x, block_x>>>(points_hessian, points_hessian_buff, valid_points_num, 18, 6);

	computePointDerivatives<<<grid_x, block_x>>>(x_, y_, z_, points_number_,
													valid_points, valid_points_num,
													dj_ang_a_, dj_ang_b_, dj_ang_c_, dj_ang_d_,
													dj_ang_e_, dj_ang_f_, dj_ang_g_, dj_ang_h_,
													dh_ang_a2_, dh_ang_a3_, dh_ang_b2_, dh_ang_b3_, dh_ang_c2_,
													dh_ang_c3_, dh_ang_d1_, dh_ang_d2_, dh_ang_d3_, dh_ang_e1_,
													dh_ang_e2_, dh_ang_e3_, dh_ang_f1_, dh_ang_f2_, dh_ang_f3_,
													points_gradient, points_hessian, compute_hessian);

	computeDerivative<<<grid_x, block_x>>>(trans_x, trans_y, trans_z, points_num,
											valid_points, voxel_id, valid_points_num,
											voxel_list, gauss_d1_, gauss_d2_,
											points_gradient, points_hessian,
											gradients_list, hessians_list,
											score, compute_hessian);
	checkCudaErrors(cudaDeviceSynchronize());

	int full_size = valid_points_num;
	int half_size = (full_size - 1) / 2 + 1;

	while (full_size > 1) {
		block_x = (half_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_size;
		grid_x = (half_size - 1) / block_x + 1;

		matrixSum<<<grid_x, block_x>>>(gradients_list, full_size, half_size);
		matrixSum<<<grid_x, block_x>>>(hessians_list, full_size, half_size);
		sumScore<<<grid_x, block_x>>>(score, full_size, half_size);

		full_size = half_size;
		half_size = (full_size - 1) / 2 + 1;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	MatrixDevice dscore_g(6, 1), dhessian(6, 6);
	MatrixHost hscore_g(6, 1), hhessian(6, 6);

	checkCudaErrors(cudaMemcpy(&dscore_g, gradients_list, sizeof(MatrixDevice), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&dhessian, hessians_list, sizeof(MatrixDevice), cudaMemcpyDeviceToHost));

	hscore_g.moveToHost(dscore_g);
	hhessian.moveToHost(dhessian);

	for (int i = 0; i < 6; i++) {
		score_gradient(i, 0) = hscore_g(i, 0);
	}

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			hessian(i, j) = hhessian(i, j);
		}
	}

	double score_inc;

	checkCudaErrors(cudaMemcpy(&score_inc, score, sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(gradients_list));
	checkCudaErrors(cudaFree(hessians_list));
	checkCudaErrors(cudaFree(points_gradient));
	checkCudaErrors(cudaFree(points_hessian));

	checkCudaErrors(cudaFree(gradient_buff));
	checkCudaErrors(cudaFree(hessian_buff));
	checkCudaErrors(cudaFree(points_hessian_buff));
	checkCudaErrors(cudaFree(points_gradient_buff));
	checkCudaErrors(cudaFree(score));

	return score_inc;
}

void GNormalDistributionsTransform::computeAngleDerivatives(MatrixHost pose, bool compute_hessian)
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

	dj_ang_a_ = MatrixDevice(3, 1);
	dj_ang_b_ = MatrixDevice(3, 1);
	dj_ang_c_ = MatrixDevice(3, 1);
	dj_ang_d_ = MatrixDevice(3, 1);
	dj_ang_e_ = MatrixDevice(3, 1);
	dj_ang_f_ = MatrixDevice(3, 1);
	dj_ang_g_ = MatrixDevice(3, 1);
	dj_ang_h_ = MatrixDevice(3, 1);

	j_ang_a_.moveToGpu(dj_ang_a_);
	j_ang_b_.moveToGpu(dj_ang_b_);
	j_ang_c_.moveToGpu(dj_ang_c_);
	j_ang_d_.moveToGpu(dj_ang_d_);
	j_ang_e_.moveToGpu(dj_ang_e_);
	j_ang_f_.moveToGpu(dj_ang_f_);
	j_ang_g_.moveToGpu(dj_ang_g_);
	j_ang_h_.moveToGpu(dj_ang_h_);

	if (compute_hessian) {
		h_ang_a2_ = MatrixHost(3, 1);
		h_ang_a3_ = MatrixHost(3, 1);
		h_ang_b2_ = MatrixHost(3, 1);
		h_ang_b3_ = MatrixHost(3, 1);
		h_ang_c2_ = MatrixHost(3, 1);
		h_ang_c3_ = MatrixHost(3, 1);
		h_ang_d1_ = MatrixHost(3, 1);
		h_ang_d2_ = MatrixHost(3, 1);
		h_ang_d3_ = MatrixHost(3, 1);
		h_ang_e1_ = MatrixHost(3, 1);
		h_ang_e2_ = MatrixHost(3, 1);
		h_ang_e3_ = MatrixHost(3, 1);
		h_ang_f1_ = MatrixHost(3, 1);
		h_ang_f2_ = MatrixHost(3, 1);
		h_ang_f3_ = MatrixHost(3, 1);


		h_ang_a2_(0) = -cx * sz - sx * sy * cz;
		h_ang_a2_(1) = -cx * cz + sx * sy * sz;
		h_ang_a2_(2) = sx * cy;

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

		dh_ang_a2_ = MatrixDevice(3, 1);
		dh_ang_a3_ = MatrixDevice(3, 1);
		dh_ang_b2_ = MatrixDevice(3, 1);
		dh_ang_b3_ = MatrixDevice(3, 1);
		dh_ang_c2_ = MatrixDevice(3, 1);
		dh_ang_c3_ = MatrixDevice(3, 1);
		dh_ang_d1_ = MatrixDevice(3, 1);
		dh_ang_d2_ = MatrixDevice(3, 1);
		dh_ang_d3_ = MatrixDevice(3, 1);
		dh_ang_e1_ = MatrixDevice(3, 1);
		dh_ang_e2_ = MatrixDevice(3, 1);
		dh_ang_e3_ = MatrixDevice(3, 1);
		dh_ang_f1_ = MatrixDevice(3, 1);
		dh_ang_f2_ = MatrixDevice(3, 1);
		dh_ang_f3_ = MatrixDevice(3, 1);

		h_ang_a2_.moveToGpu(dh_ang_a2_);
		h_ang_a3_.moveToGpu(dh_ang_a3_);
		h_ang_b2_.moveToGpu(dh_ang_b2_);
		h_ang_b3_.moveToGpu(dh_ang_b3_);
		h_ang_c2_.moveToGpu(dh_ang_c2_);
		h_ang_c2_.moveToGpu(dh_ang_c3_);
		h_ang_d1_.moveToGpu(dh_ang_d1_);
		h_ang_d2_.moveToGpu(dh_ang_d2_);
		h_ang_d3_.moveToGpu(dh_ang_d3_);
		h_ang_e1_.moveToGpu(dh_ang_e1_);
		h_ang_e2_.moveToGpu(dh_ang_e2_);
		h_ang_e3_.moveToGpu(dh_ang_e3_);
		h_ang_f1_.moveToGpu(dh_ang_f1_);
		h_ang_f2_.moveToGpu(dh_ang_f2_);
		h_ang_f3_.moveToGpu(dh_ang_f3_);
	}

}




extern "C" __global__ void gpuTransform(float *in_x, float *in_y, float *in_z,
										float *trans_x, float *trans_y, float *trans_z,
										int point_num, MatrixDevice transform)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float x, y, z;

	for (int i = idx; i < point_num; i += stride) {
		x = in_x[i];
		y = in_y[i];
		z = in_z[i];
		trans_x[i] = transform(0, 0) * x + transform(0, 1) * y + transform(0, 2) * z + transform(0, 3);
		trans_y[i] = transform(1, 0) * x + transform(1, 1) * y + transform(1, 2) * z + transform(1, 3);
		trans_z[i] = transform(2, 0) * x + transform(2, 1) * y + transform(2, 2) * z + transform(2, 3);
	}
}

void GNormalDistributionsTransform::transformPointCloud(float *in_x, float *in_y, float *in_z,
														float *trans_x, float *trans_y, float *trans_z,
														int points_number, Eigen::Matrix<float, 4, 4> transform)
{
	Eigen::Transform<float, 3, Eigen::Affine> t(transform);

	MatrixHost htrans(3, 4);
	MatrixDevice dtrans(3, 4);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			htrans(i, j) = t(i, j);
		}
	}

	htrans.moveToGpu(dtrans);

	if (points_number > 0) {
		int block_x = (points_number <= BLOCK_SIZE_X) ? points_number : BLOCK_SIZE_X;
		int grid_x = (points_number - 1) / block_x + 1;

		gpuTransform<<<grid_x, block_x >>>(in_x, in_y, in_z, trans_x, trans_y, trans_z, points_number, dtrans);
		checkCudaErrors(cudaDeviceSynchronize());
	}
}

double GNormalDistributionsTransform::computeStepLengthMT(const Eigen::Matrix<double, 6, 1> &x, Eigen::Matrix<double, 6, 1> &step_dir,
															double step_init, double step_max, double step_min, double &score,
															Eigen::Matrix<double, 6, 1> &score_gradient, Eigen::Matrix<double, 6, 6> &hessian,
															float *trans_x, float *trans_y, float *trans_z, int points_num)
{
	double phi_0 = -score;
	double d_phi_0 = -(score_gradient.dot(step_dir));

	Eigen::Matrix<double, 6, 1> x_t;

	if (d_phi_0 >= 0) {
		if (d_phi_0 == 0)
			return 0;
		else {
			d_phi_0 *= -1;
			step_dir *= -1;
		}
	}

	int max_step_iterations = 10;
	int step_iterations = 0;


	double mu = 1.e-4;
	double nu = 0.9;
	double a_l = 0, a_u = 0;

	double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
	double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

	double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
	double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

	bool interval_converged = (step_max - step_min) > 0, open_interval = true;

	double a_t = step_init;
	a_t = std::min(a_t, step_max);
	a_t = std::max(a_t, step_min);

	x_t = x + step_dir * a_t;

	final_transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ())).matrix();

	transformPointCloud(x_, y_, z_, trans_x, trans_y, trans_z, points_num, final_transformation_);

	score = computeDerivatives(score_gradient, hessian, trans_x, trans_y, trans_z, points_num, x_t, true);

	double phi_t = -score;
	double d_phi_t = -(score_gradient.dot(step_dir));
	double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
	double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

	while (!interval_converged && step_iterations < max_step_iterations && !(psi_t <= 0 && d_phi_t <= -nu * d_phi_0)) {
		if (open_interval) {
			a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
		} else {
			a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
		}

		a_t = (a_t < step_max) ? a_t : step_max;
		a_t = (a_t > step_min) ? a_t : step_min;

		x_t = x + step_dir * a_t;

		final_transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ())).matrix();

		transformPointCloud(x_, y_, z_, trans_x, trans_y, trans_z, points_num, final_transformation_);

		score = computeDerivatives(score_gradient, hessian, trans_x, trans_y, trans_z, points_num, x_t, false);

		phi_t -= score;
		d_phi_t -= (score_gradient.dot(step_dir));
		psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
		d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

		if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
			open_interval = false;

			f_l += phi_0 - mu * d_phi_0 * a_l;
			g_l += mu * d_phi_0;

			f_u += phi_0 - mu * d_phi_0 * a_u;
			g_u += mu * d_phi_0;
		}

		if (open_interval) {
			interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
		} else {
			interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
		}

		step_iterations++;
	}

	if (step_iterations)
		computeHessian(hessian, trans_x, trans_y, trans_z, points_num, x_t);

	return a_t;
}


//Copied from ndt.hpp
double GNormalDistributionsTransform::trialValueSelectionMT (double a_l, double f_l, double g_l,
															double a_u, double f_u, double g_u,
															double a_t, double f_t, double g_t)
{
	// Case 1 in Trial Value Selection [More, Thuente 1994]
	if (f_t > f_l) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
		// Equation 2.4.2 [Sun, Yuan 2006]
		double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

		if (std::fabs (a_c - a_l) < std::fabs (a_q - a_l))
		  return (a_c);
		else
		  return (0.5 * (a_q + a_c));
	}
	// Case 2 in Trial Value Selection [More, Thuente 1994]
	else if (g_t * g_l < 0) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		if (std::fabs (a_c - a_t) >= std::fabs (a_s - a_t))
		  return (a_c);
		else
		  return (a_s);
	}
	// Case 3 in Trial Value Selection [More, Thuente 1994]
	else if (std::fabs (g_t) <= std::fabs (g_l)) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		double a_t_next;

		if (std::fabs (a_c - a_t) < std::fabs (a_s - a_t))
		  a_t_next = a_c;
		else
		  a_t_next = a_s;

		if (a_t > a_l)
		  return (std::min (a_t + 0.66 * (a_u - a_t), a_t_next));
		else
		  return (std::max (a_t + 0.66 * (a_u - a_t), a_t_next));
	}
	// Case 4 in Trial Value Selection [More, Thuente 1994]
	else {
		// Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
		double w = std::sqrt (z * z - g_t * g_u);
		// Equation 2.4.56 [Sun, Yuan 2006]
		return (a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w));
	}
}

//Copied from ndt.hpp
double GNormalDistributionsTransform::updateIntervalMT (double &a_l, double &f_l, double &g_l,
														double &a_u, double &f_u, double &g_u,
														double a_t, double f_t, double g_t)
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
	if (f_t > f_l) {
		a_u = a_t;
		f_u = f_t;
		g_u = g_t;
		return (false);
	}
	// Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
	else if (g_t * (a_l - a_t) > 0) {
		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
	else if (g_t * (a_l - a_t) < 0) {
		a_u = a_l;
		f_u = f_l;
		g_u = g_l;

		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Interval Converged
	else
		return (true);
}

extern "C" __global__ void updateHessian(float *trans_x, float *trans_y, float *trans_z, int points_num,
											int *valid_points, int *voxel_id, int valid_points_num,
											GVoxel *grid, float gauss_d1, float gauss_d2,
											MatrixDevice *point_gradients, MatrixDevice *point_hessians, MatrixDevice *hessians)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = valid_points[i];

		//Transformed coordinates
		float t_x = trans_x[pid];
		float t_y = trans_y[pid];
		float t_z = trans_z[pid];

		MatrixDevice pg = point_gradients[i];		//3x6 Matrix
		MatrixDevice ph = point_hessians[i];		//18x6 Matrix
		MatrixDevice h = hessians[i];				//6x6 Matrix

		double score_inc = 0;

		for ( int vid = voxel_id[i]; vid < voxel_id[i + 1]; vid++) {
			GVoxel *voxel = grid + vid;
			MatrixDevice centroid = voxel->centroid();
			MatrixDevice icov = voxel->inverseCovariance();	//3x3 matrix

			float cov_dxd_pi_x, cov_dxd_pi_y, cov_dxd_pi_z;

			t_x -= centroid(0);
			t_y -= centroid(1);
			t_z -= centroid(2);

			double e_x_cov_x = expf(-gauss_d2 * ((t_x * icov(0, 0) + t_y * icov(1, 0) + t_z * icov(2, 0)) * t_x
					+ ((t_x * icov(0, 1) + t_y * icov(1, 1) + t_z * icov(2, 1)) * t_y)
					+ ((t_x * icov(0, 2) + t_y * icov(1, 2) + t_z * icov(2, 2)) * t_z)) / 2);
			score_inc += -gauss_d1 * e_x_cov_x;

			e_x_cov_x *= gauss_d2;

			e_x_cov_x *= gauss_d1;

			for (int n = 0; n < 6; n++) {
				cov_dxd_pi_x = icov(0, 0) * pg(0, n) + icov(0, 1) * pg(1, n) + icov(0, 2) * pg(2, n);
				cov_dxd_pi_y = icov(1, 0) * pg(0, n) + icov(1, 1) * pg(1, n) + icov(1, 2) * pg(2, n);
				cov_dxd_pi_z = icov(2, 0) * pg(0, n) + icov(2, 1) * pg(1, n) + icov(2, 2) * pg(2, n);

				//Compute hessian
				for (int p = 0; p < h.cols(); p++) {
					h(n, p) += e_x_cov_x * (-gauss_d2 * (t_x * cov_dxd_pi_x + t_y * cov_dxd_pi_y + t_z * cov_dxd_pi_z) *
								(t_x * (icov(0, 0) * pg(0, p) + icov(0, 1) * pg(1, p) + icov(0, 2) * pg(2, p))
								+ t_y * (icov(1, 0) * pg(0, p) + icov(1, 1) * pg(1, p) + icov(1, 2) * pg(2, p))
								+ t_z * (icov(2, 0) * pg(0, p) + icov(2, 1) * pg(1, p) + icov(2, 2) * pg(2, p)))
								+ (t_x * (icov(0, 0) * ph(3 * n, p) + icov(0, 1) * ph(3 * n + 1, p) + icov(0, 2) * ph(3 * n + 2, p))
								+ t_y * (icov(1, 0) * ph(3 * n, p) + icov(1, 1) * ph(3 * n + 1, p) + icov(1, 2) * ph(3 * n + 2, p))
								+ t_z * (icov(2, 0) * ph(3 * n, p) + icov(2, 1) * ph(3 * n + 1, p) + icov(2, 2) * ph(3 * n + 2, p)))
								+ (pg(0, p) * cov_dxd_pi_x + pg(1, p) * cov_dxd_pi_y + pg(2, p) * cov_dxd_pi_z));
				}
			}
		}
	}
}

void GNormalDistributionsTransform::computeHessian (Eigen::Matrix<double, 6, 6> &hessian, float *trans_x, float *trans_y, float *trans_z, int points_num, Eigen::Matrix<double, 6, 1> &p)
{
	//Radius Search
	voxel_grid_.radiusSearch(trans_x, trans_y, trans_z, points_num, resolution_, INT_MAX);

	int *valid_points = voxel_grid_.getValidPoints();

	int *voxel_id = voxel_grid_.getVoxelIds();

	int search_size = voxel_grid_.getSearchResultSize();

	int valid_points_num = voxel_grid_.getValidPointsNum();

	GVoxel *voxel_list = voxel_grid_.getVoxelList();

	int voxel_num = voxel_grid_.getVoxelNum();

	float max_x = voxel_grid_.getMaxX();
	float max_y = voxel_grid_.getMaxY();
	float max_z = voxel_grid_.getMaxZ();

	float min_x = voxel_grid_.getMinX();
	float min_y = voxel_grid_.getMinY();
	float min_z = voxel_grid_.getMinZ();

	float voxel_x = voxel_grid_.getVoxelX();
	float voxel_y = voxel_grid_.getVoxelY();
	float voxel_z = voxel_grid_.getVoxelZ();

	int max_b_x = voxel_grid_.getMaxBX();
	int max_b_y = voxel_grid_.getMaxBY();
	int max_b_z = voxel_grid_.getMaxBZ();

	int min_b_x = voxel_grid_.getMinBX();
	int min_b_y = voxel_grid_.getMinBY();
	int min_b_z = voxel_grid_.getMinBZ();

	//Update score gradient and hessian matrix
	MatrixDevice *hessians_list, *points_gradient, *points_hessian;

	checkCudaErrors(cudaMalloc(&hessians_list, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&points_gradient, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&points_hessian, sizeof(MatrixDevice) * valid_points_num));

	double *hessian_buff, *points_gradient_buff, *points_hessian_buff;

	checkCudaErrors(cudaMalloc(&hessian_buff, sizeof(double) * valid_points_num * 6 * 6));
	checkCudaErrors(cudaMalloc(&points_gradient_buff, sizeof(double) * valid_points_num * 3 * 6));
	checkCudaErrors(cudaMalloc(&points_hessian_buff, sizeof(double) * valid_points_num * 18 * 6));

	int block_x = (valid_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : valid_points_num;
	int grid_x = (valid_points_num - 1) / block_x + 1;

	matrixListInit<<<grid_x, block_x>>>(hessians_list, hessian_buff, valid_points_num, 6, 6);
	matrixListInit<<<grid_x, block_x>>>(points_gradient, points_gradient_buff, valid_points_num, 3, 6);
	matrixListInit<<<grid_x, block_x>>>(points_hessian, points_hessian_buff, valid_points_num, 18, 6);

	computePointDerivatives<<<grid_x, block_x>>>(x_, y_, z_, points_number_,
													valid_points, valid_points_num,
													dj_ang_a_, dj_ang_b_, dj_ang_c_, dj_ang_d_,
													dj_ang_e_, dj_ang_f_, dj_ang_g_, dj_ang_h_,
													dh_ang_a2_, dh_ang_a3_, dh_ang_b2_, dh_ang_b3_, dh_ang_c2_,
													dh_ang_c3_, dh_ang_d1_, dh_ang_d2_, dh_ang_d3_, dh_ang_e1_,
													dh_ang_e2_, dh_ang_e3_, dh_ang_f1_, dh_ang_f2_, dh_ang_f3_,
													points_gradient, points_hessian, true);

	updateHessian<<<grid_x, block_x>>>(trans_x, trans_y, trans_z, points_num,
											valid_points, voxel_id, valid_points_num,
											voxel_list, gauss_d1_, gauss_d2_,
											points_gradient, points_hessian, hessians_list);
	checkCudaErrors(cudaDeviceSynchronize());

	int full_size = valid_points_num;
	int half_size = (full_size - 1) / 2 + 1;

	while (full_size > 1) {
		block_x = (half_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_size;
		grid_x = (half_size - 1) / block_x + 1;

		matrixSum<<<grid_x, block_x>>>(hessians_list, full_size, half_size);

		full_size = half_size;
		half_size = (full_size - 1) / 2 + 1;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	MatrixDevice dhessian(6, 6);
	MatrixHost hhessian(6, 6);

	checkCudaErrors(cudaMemcpy(&dhessian, hessians_list, sizeof(MatrixDevice), cudaMemcpyDeviceToHost));

	hhessian.moveToHost(dhessian);

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			hessian(i, j) = hhessian(i, j);
		}
	}

	checkCudaErrors(cudaFree(hessians_list));
	checkCudaErrors(cudaFree(points_gradient));
	checkCudaErrors(cudaFree(points_hessian));

	checkCudaErrors(cudaFree(hessian_buff));
	checkCudaErrors(cudaFree(points_hessian_buff));
	checkCudaErrors(cudaFree(points_gradient_buff));
}

}
