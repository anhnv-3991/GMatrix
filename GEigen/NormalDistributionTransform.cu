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

extern "C" __global__ void matrixListInit(MatrixDevice *matrix, float *matrix_buff, int matrix_num, int rows, int cols)
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

double GNormalDistributionTransform::computeDerivatives(MatrixDevice score_gradient, MatrixDevice hessian,
														float *source_x, float *source_y, float *source_z,
														int points_num,
														MatrixDevice p, bool compute_hessian)
{
	//Compute Angle Derivatives
	computeAngleDerivatives(p, compute_hessian);
	//Radius Search
	voxel_grid_.radiusSearch(source_x, source_y, source_z, points_num, resolution_, INT_MAX);

	int *valid_points = voxel_grid_.getValidPoints();

	int *neighbor_id = voxel_grid_.getNeighborIds();

	int *voxel_id = voxel_grid_.getVoxelIds();

	int search_size = voxel_grid_.getSearchResultSize();

	int valid_points_num = voxel_grid_.getValidPointsNum();

	GVoxel *voxel_list = voxel_grid_.getVoxelList();

	int voxel_x = voxel_grid_.getVoxelX();
	int voxel_y = voxel_grid_.getVoxelY();
	int voxel_z = voxel_grid_.getVoxelZ();

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

	int vgrid_x = voxel_grid_.getVgridX();
	int vgrid_y = voxel_grid_.getVgridY();
	int vgrid_z = voxel_grid_.getVgridZ();

	//Update score gradient and hessian matrix
	MatrixDevice *gradient_list, *hessian_list, *points_gradient, *points_hessian;

	checkCudaErrors(cudaMalloc(&gradient_list, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&hessian_list, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&points_gradient, sizeof(MatrixDevice) * valid_points_num));
	checkCudaErrors(cudaMalloc(&points_hessian, sizeof(MatrixDevice) * valid_points_num));

	float *gradient_buff, *hessian_buff, *points_gradient_buff, *poinst_hessian_buff, *score;

	checkCudaErrors(cudaMalloc(&gradient_buff, sizeof(float) * valid_points_num * 6));
	checkCudaErrors(cudaMalloc(&hessian_buff, sizeof(float) * valid_points_num * 6 * 6));
	checkCudaErrors(cudaMalloc(&points_gradient_buff, sizeof(float) * valid_points_num * 3 * 6));
	checkCudaErrors(cudaMalloc(&points_hessian_buff, sizeof(float) * valid_points_num * 18 * 6));
	checkCudaErrors(cudaMalloc(&score, sizeof(float) * valid_points_num));

	int block_x = (valid_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : valid_points_num;
	int grid_x = (valid_points_num - 1) / block_x + 1;

	matrixListInit<<<grid_x, block_x>>>(gradient_list, gradient_buff, valid_points_num, 1, 6);
	matrixListInit<<<grid_x, block_x>>>(hessian_list, hessian_buff, valid_points_num, 6, 6);
	matrixListInit<<<grid_x, block_x>>>(points_gradient, points_gradient_buff, valid_points_num, 3, 6);
	matrixListInit<<<grid_x, block_x>>>(poinst_hessian, points_hessian_buff, valid_points_num, 18, 6);

	computeDerivative<<<grid_x, block_x>>>(source_x, source_y, source_z, points_num,
											valid_points, voxel_id, valid_points_num,
											voxel_list, vgrid_x, vgrid_y, vgrid_z,
											dj_ang_a_, dj_ang_b_, dj_ang_c_, dj_ang_d_,
											dj_ang_e_, dj_ang_f_, dj_ang_g_, dj_ang_h_,
											dh_ang_a2_, dh_ang_a3_, dh_ang_b2_, dh_ang_b3_, dh_ang_c2_,
											dh_ang_c3_, dh_ang_d1_, dh_ang_d2_, dh_ang_d3_, dh_ang_e1_,
											dh_ang_e2_, dh_ang_e3_, dh_ang_f1_, dh_ang_f2_, dh_ang_f3_,
											(float)gauss_d1_, (float)gauss_d2_,
											point_gradients, point_hessians, gradients_list, hessians_list,
											score);
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

extern "C" __global__ void computeDerivative(float *x, float *y, float *z, int points_num,
												int *valid_points, int *voxel_id, int valid_points_num,
												GVoxel *grid, int vgrid_x, int vgrid_y, int vgrid_z,
												MatrixDevice j_ang_a, MatrixDevice j_ang_b, MatrixDevice j_ang_c, MatrixDevice j_ang_d,
												MatrixDevice j_ang_e, MatrixDevice j_ang_f, MatrixDevice j_ang_g, MatrixDevice j_ang_h,
												MatrixDevice h_ang_a2_, MatrixDevice h_ang_a3_, MatrixDevice h_ang_b2_, MatrixDevice h_ang_b3_, MatrixDevice h_ang_c2_,
												MatrixDevice h_ang_c3_, MatrixDevice h_ang_d1_, MatrixDevice h_ang_d2_, MatrixDevice h_ang_d3_, MatrixDevice h_ang_e1_,
												MatrixDevice h_ang_e2_, MatrixDevice h_ang_e3_, MatrixDevice h_ang_f1_, MatrixDevice h_ang_f2_, MatrixDevice h_ang_f3_,
												float gauss_d1, float gauss_d2,
												MatrixDevice *point_gradients, MatrixDevice *point_hessians, MatrixDevice *score_gradients, MatrixDevice *hessians,
												float *score)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = point_id[i];

		float t_x = x[pid];
		float t_y = y[pid];
		float t_z = z[pid];

		MatrixDevice pg = point_gradients[i];		//3x6 Matrix
		MatrixDevice ph = point_hessians[i];		//18x6 Matrix
		MatrixDevice sg = score_gradients[i];		//6x1 Matrix
		MatrixDevice h = hessians[i];				//6x6 Matrix

		double score_inc = 0;

		for ( int vid = voxel_id[i]; vid < voxel_id[i + 1]; vid++) {
			GVoxel *voxel = grid[vid];
			MatrixDevice centroid = voxel->centroid();

			pg(1, 3) = t_x * j_ang_a(0) + t_y * j_ang_a(1) + t_z * t_ang_a(2);
			pg(2, 3) = t_x * j_ang_b(0) + t_y * j_ang_b(1) + t_z * t_ang_b(2);
			pg(0, 4) = t_x * j_ang_c(0) + t_y * j_ang_c(1) + t_z * j_ang_c(2);
			pg(1, 4) = t_x * j_ang_d(0) + t_y * j_ang_d(1) + t_z * j_ang_d(2);
			pg(2, 4) = t_x * j_ang_e(0) + t_y * j_ang_e(1) + t_z * j_ang_e(2);
			pg(0, 5) = t_x * j_ang_f(0) + t_y * j_ang_f(1) + t_z * j_ang_f(2);
			pg(1, 5) = t_x * j_ang_g(0) + t_y * j_ang_g(1) + t_z * j_ang_g(2);
			pg(2, 5) = t_x * j_ang_h(0) + t_y * j_ang_h(1) + t_z * j_ang_h(2);

			ph(9, 3) = 0;
			ph(10, 3) = t_x * h_ang_a2(0) + t_y * h_ang_a2(1) + t_z * h_ang_a2(2);
			ph(11, 3) = t_x * h_ang_a3(0) + t_y * h_ang_a3(1) + t_z * h_ang_a3(2);

			ph(12, 3) = ph(9, 4) = 0;
			ph(13, 3) = ph(10, 4) = t_x * h_ang_b2(0) + t_y * h_ang_b2(1) + t_z * h_ang_b2(2);
			ph(14, 3) = ph(11, 4) = t_x * h_ang_b3(0) + t_y * h_ang_b3(1) + t_z * h_ang_b3(2);

			ph(15, 3) = 0;
			ph(16, 3) = ph(9, 5) = t_x * h_ang_c2(0) + t_y * h_ang_c2(1) + t_z * h_ang_c2(2);
			ph(17, 3) = ph(10, 5) = t_x * h_ang_c3(0) + t_y * h_ang_c3(1) + t_z * h_ang_c3(2);

			ph(12, 4) = t_x * h_ang_d1(0) + t_y * h_ang_d1(1) + t_z * h_ang_d1(2);
			ph(13, 4) = t_x * h_ang_d2(0) + t_y * h_ang_d2(1) + t_z * h_ang_d2(2);
			ph(14, 4) = t_x * h_ang_d3(0) + t_y * h_ang_d3(1) + t_z * h_ang_d3(2);

			ph(15, 4) = ph(12, 5) = t_x * h_ang_e1(0) + t_y * h_ang_e1(1) + t_z * h_ang_e1(2);
			ph(16, 4) = ph(13, 5) = t_x * h_ang_e2(0) + t_y * h_ang_e2(1) + t_z * h_ang_e2(2);
			ph(17, 4) = ph(14, 5) = t_x * h_ang_e3(0) + t_y * h_ang_e3(1) + t_z * h_ang_e3(2);

			ph(15, 5) = t_x * h_ang_f1(0) + t_y * h_ang_f1(1) + t_z * h_ang_f1(2);
			ph(16, 5) = t_x * h_ang_f2(0) + t_y * h_ang_f2(1) + t_z * h_ang_f2(2);
			ph(17, 5) = t_x * h_ang_f3(0) + t_y * h_ang_f3(1) + t_z * h_ang_f3(2);

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

				sg(n) += (t_x * cov_dxd_pi_x + t_y * cov_dxd_pi_y + t_z * cov_dxd_pi_z) * e_x_cov_x;

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

/* Compute sum of a list of a matrixes */
extern "C" __global__ void matrixSum(MatrixDevice *matrix_list, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride) {
		MatrixDevice left = matrix_list[i];
		MatrixDevice right = (i + half_size < full_size) ? matrix_list[i + half_size] : MatrixDevice();

		if (!right.isEmpty()) {
			for (int j = 0; j < left->rows(); j++) {
				for (int k = 0; k < left->cols(); k++) {
					left(j, k) += right(j, k);
				}
			}
		}
	}
}

extern "C" __global__ void sumScore(float *score, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride) {
		score[i] += (i + half_size < full_size) ? score[i + half_size] : 0;
	}
}


}
