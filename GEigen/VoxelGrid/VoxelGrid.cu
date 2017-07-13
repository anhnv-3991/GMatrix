#include "VoxelGrid.h"
#include "../debug.h"
#include "../common.h"
#include <math.h>
#include <limits>
#include <eigen3/Eigen/Eigenvalues>

namespace gpu {

extern "C" __global__ void initVoxelGrid(GVoxel *voxel_grid, int vgrid_x, int vgrid_y, int vgrid_z,
											float min_xy, float min_yz, float min_zx,
											float voxel_x, float voxel_y, float voxel_z,
											float *centroid_buff, float *cov_buff, float *inverse_cov_buff)
{
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_z = threadIdx.z + blockIdx.z * blockDim.z;

	if (id_x < vgrid_x && id_y < vgrid_y && id_z < vgrid_z) {
		int vgrid_id = id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y;

		GVoxel *voxel = voxel_grid + vgrid_id;

		
		voxel->minXY() = min_xy + id_z * voxel_z;
		voxel->minYZ() = min_yz + id_x * voxel_x;
		voxel->minZX() = min_zx + id_y * voxel_y;

		voxel->maxXY() = min_xy + id_z * voxel_z + voxel_z;
		voxel->maxYZ() = min_yz + id_x * voxel_x + voxel_x;
		voxel->maxZX() = min_zx + id_y * voxel_y + voxel_y;

		voxel->centroid() = MatrixDevice(1, 3, vgrid_x * vgrid_y * vgrid_z, centroid_buff);
		voxel->covariance() = MatrixDevice(3, 3, vgrid_x * vgrid_y * vgrid_z, cov_buff);
		voxel->inverseCovariance() = MatrixDevice(3, 3, vgrid_x * vgrid_y * vgrid_z, inverse_cov_buff);

		voxel->pointNum() = 0;
	}
}

void GVoxelGrid::initialize()
{
	float *centroid, *covariance, *inverse_covariance;

	checkCudaErrors(cudaMalloc(&centroid, sizeof(float) * 3 * vgrid_x_ * vgrid_y_ * vgrid_z_));
	checkCudaErrors(cudaMalloc(&covariance, sizeof(float) * 9 * vgrid_x_ * vgrid_y_ * vgrid_z_));
	checkCudaErrors(cudaMalloc(&inverse_covariance, sizeof(float) * 9 * vgrid_x_ * vgrid_y_ * vgrid_z_));

	int block_x = (vgrid_x_ > BLOCK_X) ? BLOCK_X : vgrid_x_;
	int block_y = (vgrid_y_ > BLOCK_Y) ? BLOCK_Y : vgrid_y_;
	int block_z = (vgrid_z_ > BLOCK_Z) ? BLOCK_Z : vgrid_z_;

	int grid_x = (vgrid_x_ - 1) / block_x + 1;
	int grid_y = (vgrid_y_ - 1) / block_y + 1;
	int grid_z = (vgrid_z_ - 1) / block_z + 1;

	dim3 block(block_x, block_y, block_z);
	dim3 grid(grid_x, grid_y, grid_z);

	initVoxelGrid<<<grid, block>>>(global_voxel_,
									vgrid_x_, vgrid_y_, vgrid_z_,
									min_xy_, min_yz_, min_zx_,
									voxel_x_, voxel_y_, voxel_z_,
									centroid, covariance, inverse_covariance);
	checkCudaErrors(cudaDeviceSynchronize());
}

__device__ int voxelId(float x, float y, float z,
						float voxel_x, float voxel_y, float voxel_z,
						int min_b_x, int min_b_y, int min_b_z,
						int vgrid_x, int vgrid_y, int vgrid_z)
{
	int id_x = static_cast<int>(floor(x / voxel_x) - static_cast<float>(min_b_x));
	int id_y = static_cast<int>(floor(y / voxel_y) - static_cast<float>(min_b_y));
	int id_z = static_cast<int>(floor(z / voxel_z) - static_cast<float>(min_b_z));

	return (id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y);
}

/* First pass: insert points to voxels
 * Number of points, coordinate sum, and initial covariance 
 * matrix in a voxel is calculated by atomicAdd.
 */
extern "C" __global__ void insertPoints(float *x, float *y, float *z, int points_num,
										GVoxel *voxel_grid, int vgrid_x, int vgrid_y, int vgrid_z,
										float voxel_x, float voxel_y, float voxel_z,
										int min_b_x, int min_b_y, int min_b_z)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < points_num; i += stride) {
		float t_x = x[i];
		float t_y = y[i];
		float t_z = z[i];
		int voxel_id = voxelId(t_x, t_y, t_z, voxel_x, voxel_y, voxel_z, min_b_x, min_b_y, min_b_z, vgrid_x, vgrid_y, vgrid_z);

		GVoxel *voxel = voxel_grid + voxel_id;

		MatrixDevice centr = voxel->centroid();
		MatrixDevice cov = voxel->covariance();

		atomicAdd(voxel->pointNumAddress(), 1);	
		
		atomicAdd(centr.cellAddr(0), t_x);
		atomicAdd(centr.cellAddr(1), t_y);
		atomicAdd(centr.cellAddr(2), t_z);

		atomicAdd(cov.cellAddr(0, 0), t_x * t_x);
		atomicAdd(cov.cellAddr(0, 1), t_x * t_y);
		atomicAdd(cov.cellAddr(0, 2), t_x * t_z);
		atomicAdd(cov.cellAddr(1, 0), t_y * t_x);
		atomicAdd(cov.cellAddr(1, 1), t_y * t_y);
		atomicAdd(cov.cellAddr(1, 2), t_y * t_z);
		atomicAdd(cov.cellAddr(2, 0), t_z * t_x);
		atomicAdd(cov.cellAddr(2, 1), t_z * t_y);
		atomicAdd(cov.cellAddr(2, 2), t_z * t_z);
	}
}


/* Second pass: update coordinate mean (centroid) and 
 * covariance matrix of each cell
 */
extern "C" __global__ void updateVoxelCentroid(GVoxel *voxel_grid, int voxel_num)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int vid = index; vid < voxel_num; vid += stride) {
		GVoxel *node = voxel_grid + vid;
		int points_num = node->pointNum();
		
		if (points_num == 0)
			return;

		MatrixDevice centr = node->centroid();
		MatrixDevice cov = node->covariance();
		MatrixDevice icov = node->inverseCovariance();

		centr /= points_num;
		cov /= points_num;
		cov(0, 0) -= centr(0) * centr(0);
		cov(0, 1) -= centr(0) * centr(1);
		cov(0, 2) -= centr(0) * centr(2);
		cov(1, 0) = cov(0, 1);
		cov(1, 1) -= centr(1) * centr(1);
		cov(1, 2) -= centr(1) * centr(2);
		cov(2, 0) = cov(0, 2);
		cov(2, 1) = cov(1, 2);
		cov(2, 2) -= centr(2) * centr(2);

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
		Eigen::Matrix3d eigen_val;
		Eigen::Vector3d pt_sum;
		Eigen::Matrix3d cov_mat;
		Eigen::Matrix3d eigen_vectors;
		Eigen::Matrix3d cov_mat_inverse;

		cov_mat(0, 0) = cov(0, 0);
		cov_mat(0, 1) = cov(0, 1);
		cov_mat(0, 2) = cov(0, 2);
		cov_mat(1, 0) = cov(1, 0);
		cov_mat(1, 1) = cov(1, 1);
		cov_mat(1, 2) = cov(1, 2);
		cov_mat(2, 0) = cov(2, 0);
		cov_mat(2, 1) = cov(2, 1);
		cov_mat(2, 2) = cov(2, 2);

		eigensolver.compute(cov_mat);
		eigen_val = eigensolver.eigenvalues().asDiagonal();
		eigen_vectors = eigensolver.eigenvectors();

		if (eigen_val(0,0) < 0 || eigen_val(1, 1) < 0 || eigen_val(2, 2) <= 0) {
			node->pointNum() = -1;
			return;
		}

		float min_eigen_val = eigen_val(2, 2) / 100;

		if (eigen_val(0, 0) < min_eigen_val) {
			eigen_val(0, 0) = min_eigen_val;

			if (eigen_val(1, 1) < min_eigen_val)
				eigen_val(1, 1) = min_eigen_val;

			cov_mat = eigen_vectors * eigen_val * eigen_vectors.inverse();
		}

		cov_mat_inverse = cov_mat.inverse();

		cov(0, 0) = cov_mat(0, 0);
		cov(0, 1) = cov_mat(0, 1);
		cov(0, 2) = cov_mat(0, 2);
		cov(1, 0) = cov_mat(1, 0);
		cov(1, 1) = cov_mat(1, 1);
		cov(1, 2) = cov_mat(1, 2);
		cov(2, 0) = cov_mat(2, 0);
		cov(2, 1) = cov_mat(2, 1);
		cov(2, 2) = cov_mat(2, 2);

		icov(0, 0) = cov_mat_inverse(0, 0);
		icov(0, 1) = cov_mat_inverse(0, 1);
		icov(0, 2) = cov_mat_inverse(0, 2);
		icov(1, 0) = cov_mat_inverse(1, 0);
		icov(1, 1) = cov_mat_inverse(1, 1);
		icov(1, 2) = cov_mat_inverse(1, 2);
		icov(2, 0) = cov_mat_inverse(2, 0);
		icov(2, 1) = cov_mat_inverse(2, 1);
		icov(2, 2) = cov_mat_inverse(2, 2);
	}
}

void GVoxelGrid::insertPoints()
{
	int block_x = (points_num_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_num;
	int grid_x = (points_num_ - 1) / block_x + 1;

	insertPoints<<<grid_x, block_x>>>(x_, y_, z_, points_num_,
										global_voxel_, vgrid_x_, vgrid_y_, vgrid_z_,
										voxel_x_, voxel_y_, voxel_z_,
										min_b_x_, min_b_y_, min_b_z_);
	int voxel_num = vgrid_x_ * vgrid_y_ * vgrid_z_;

	block_x = (voxel_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : voxel_num;
	grid_x = (voxel_num - 1) / block_x + 1;

	updateVoxelCentroid<<<grid_x, block_x>>>(global_voxel_, voxel_num);

	checkCudaErrors(cudaDeviceSynchronize());
}

//Input are supposed to be in device memory
void GVoxelGrid::setInput(float *x, float *y, float *z, int points_num)
{
	x_ = x;
	y_ = y;
	z_ = z;
	points_num_ = points_num;

	findBoundaries();

	checkCudaErrors(cudaMalloc(&global_voxel_, sizeof(GVoxel) * vgrid_x_ * vgrid_y_ * vgrid_z_));
	findBoundaries();
	initialize();
	insertPoints();
}

extern "C" __global__ void findMax(float *x, float *y, float *z, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride)
		x[i] = (i + half_size < full_size) ? ((x[i] >= x[i + half_size]) ? x[i] : x[i + half_size]) : x[i];
}

extern "C" __global__ void findMin(float *x, float *y, float *z, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride)
		x[i] = (i + half_size < full_size) ? ((x[i] <= x[i + half_size]) ? x[i] : x[i + half_size]) : x[i];
}


void GVoxelGrid::findBoundaries()
{
	float *max_x, *max_y, *max_z, *min_x, *min_y, *min_z;

	checkCudaErrors(cudaMalloc(&max_x, sizeof(float) * points_num_));
	checkCudaErrors(cudaMalloc(&max_y, sizeof(float) * points_num_));
	checkCudaErrors(cudaMalloc(&max_z, sizeof(float) * points_num_));
	checkCudaErrors(cudaMalloc(&min_x, sizeof(float) * points_num_));
	checkCudaErrors(cudaMalloc(&min_y, sizeof(float) * points_num_));
	checkCudaErrors(cudaMalloc(&min_z, sizeof(float) * points_num_));

	checkCudaErrors(cudaMemcpy(max_x, x_, sizeof(float) * points_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_y, y_, sizeof(float) * points_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(max_z, z_, sizeof(float) * points_num_, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpy(min_x, x_, sizeof(float) * points_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(min_y, y_, sizeof(float) * points_num_, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(min_z, z_, sizeof(float) * points_num_, cudaMemcpyDeviceToDevice));

	int points_num = points_num_;

	while (points_num > 1) {
		int half_points_num = (poinst_num - 1) / 2 + 1;
		int block_x = (half_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_points_num;
		int grid_x = (half_points_num - 1) / block_x + 1;

		findMax<<<grid_x, block_x>>>(max_x, max_y, max_z, points_num, half_points_num);
		findMin<<<grid_x, block_x>>>(min_x, min_y, min_z, points_num, half_points_num);

		points_num = half_points_num;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(&max_x_, max_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_y_, max_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_z_, max_z, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&min_x_, min_x, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_y_, min_y, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&min_z_, min_z, sizeof(float), cudaMemcpyDeviceToHost));

	max_b_x_ = static_cast<int> (floor(max_x_ / voxel_x_));
	max_b_y_ = static_cast<int> (floor(max_y_ / voxel_y_));
	max_b_z_ = static_cast<int> (floor(max_z_ / voxel_z_));

	min_b_x_ = static_cast<int> (floor(min_x_ / voxel_x_));
	min_b_y_ = static_cast<int> (floor(min_y_ / voxel_y_));
	min_b_z_ = static_cast<int> (floor(min_z_ / voxel_z_));

	vgrid_x_ = max_b_x_ - min_b_x_ + 1;
	vgrid_y_ = max_b_y_ - min_b_y_ + 1;
	vgrid_z_ = max_b_z_ - min_b_z_ + 1;
}

__device__ float squareDistance(float x, float y, float z, float a, float b, float c)
{
	return (x - a) * (x - a) + (y - b) * (y - b) + (z - c) * (z - c);
}




extern "C" __global__ void radiusSearch1(float *x, float *y, float *z, int radius, int max_nn, int points_num,
											GVoxel *grid, int vgrid_x, int vgrid_y, int vgrid_z,
											float voxel_x, float voxel_y, float voxel_z,
											int min_b_x, int min_b_y, int min_b_z,
											int max_b_x, int max_b_y, int max_b_z,
											int *max_vid_x, int *max_vid_y, int *max_vid_z,
											int *min_vid_x, int *min_vid_y, int *min_vid_z,
											int *found_voxel_num, int *valid_points)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < points_num; i += stride) {
		float t_x = x[i];
		float t_y = y[i];
		float t_z = z[i];

		int id_x = static_cast<int>(floor(x / voxel_x) - static_cast<float>(min_b_x));
		int id_y = static_cast<int>(floor(y / voxel_y) - static_cast<float>(min_b_y));
		int id_z = static_cast<int>(floor(z / voxel_z) - static_cast<float>(min_b_z));

		int max_id_x = static_cast<int>(ceilf((t_x + radius) / voxel_x) - static_cast<float>(min_b_x));
		int max_id_y = static_cast<int>(ceilf((t_y + radius) / voxel_y) - static_cast<float>(min_b_y));
		int max_id_z = static_cast<int>(ceilf((t_z + radius) / voxel_z) - static_cast<float>(min_b_z));

		int min_id_x = static_cast<int>(ceilf((t_x - radius) / voxel_x) - static_cast<float>(min_b_x));
		int min_id_y = static_cast<int>(ceilf((t_y - radius) / voxel_y) - static_cast<float>(min_b_y));
		int min_id_z = static_cast<int>(ceilf((t_z - radius) / voxel_z) - static_cast<float>(min_b_z));

		max_id_x = (max_id_x > max_b_x) ? max_b_x : max_id_x;
		max_id_y = (max_id_y > max_b_y) ? max_b_y : max_id_y;
		max_id_z = (max_id_z > max_b_z) ? max_b_z : max_id_z;

		min_id_x = (min_id_x < min_b_x) ? min_b_x : min_id_x;
		min_id_y = (min_id_y < min_b_y) ? min_b_y : min_id_y;
		min_id_z = (min_id_z < min_b_z) ? min_b_z : min_id_z;

		int nn = 0;

		for (int j = min_id_x; j <= max_id_x && nn < max_nn; j++) {
			for (int k = min_id_y; k <= max_id_y && nn < max_nn; k++) {
				for (int l = min_id_z; l <= max_id_z && nn < max_nn; l++) {
					int voxel_id = j + k * vgrid_x + l * vgrid_x * vgrid_y;
					GVoxel *voxel = grid[voxel_id];
					int point_num = voxel->pointNum();

					float centroid_x = (point_num > 0) ? voxel->centroid()(0) : FLT_MAX;
					float centroid_y = (point_num > 0) ? voxel->centroid()(1) : FLT_MAX;
					float centroid_z = (point_num > 0) ? voxel->centroid()(2) : FLT_MAX;

					nn += (squareDistance(centroid_x, centroid_y, centroid_z, t_x, t_y, t_z) <= radius * radius) ? 1 : 0;
				}
			}
		}

		found_voxel_num[i] = nn;
		valid_point[i] = (nn == 0) ? 0 : 1;

		max_vid_x[i] = max_id_x;
		max_vid_y[i] = max_id_y;
		max_vid_z[i] = max_id_z;

		min_vid_x[i] = min_id_x;
		min_vid_y[i] = min_id_y;
		min_vid_z[i] = min_id_z;
	}
}

extern "C" __global__ void collectValidPoints(int *input, int *output, int *writing_location, int size)
{
	for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index += blockDim.x * gridDim.x) {
		if (input[index] != 0) {
			output[writing_location[index]] = index;
		}
	}
}

extern "C" __global__ void radiusSearch2(float *x, float *y, float *z, int radius, int max_nn, int points_num,
											GVoxel *grid, int vgrid_x, int vgrid_y, int vgrid_z,
											int *max_vid_x, int *max_vid_y, int *max_vid_z,
											int *min_vid_x, int *min_vid_y, int *min_vid_z,
											int *found_voxel_num, int *voxel_id)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < points_num; i += stride) {
		float t_x = x[i];
		float t_y = y[i];
		float t_z = z[i];

		int max_id_x = max_vid_x[i];
		int max_id_y = max_vid_y[i];
		int max_id_z = max_vid_z[i];

		int min_id_x = min_vid_x[i];
		int min_id_y = min_vid_y[i];
		int min_id_z = min_vid_z[i];

		int nn = 0;
		int write_location = found_voxel_num[i];
		int pn = found_voxel_num[i + 1] - found_voxel_num[i];

		if (pn == 0)
			return;

		for (int j = min_id_x; j <= max_id_x && nn < max_nn; j++) {
			for (int k = min_id_y; k <= max_id_y && nn < max_nn; k++) {
				for (int l = min_id_z; l <= max_id_z && nn < max_nn; l++) {
					int voxel_id = j + k * vgrid_x + l * vgrid_x * vgrid_y;
					GVoxel *voxel = grid[voxel_id];
					int point_num = voxel->pointNum();

					float centroid_x = (point_num > 0) ? voxel->centroid()(0) : FLT_MAX;
					float centroid_y = (point_num > 0) ? voxel->centroid()(1) : FLT_MAX;
					float centroid_z = (point_num > 0) ? voxel->centroid()(2) : FLT_MAX;

					if (squareDistance(centroid_x, centroid_y, centroid_z, t_x, t_y, t_z) <= radius * radius) {
						voxel_id[write_location] = voxel_id;

						write_location++;
						nn++;
					}
				}
			}
		}
	}
}

template <typename T>
void GVoxelGrid::ExclusiveScan(T *input, int ele_num, T *sum)
{
	thrust::device_ptr<T> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

void GVoxelGrid::radiusSearch(float *qx, float *qy, float *qz, int points_num, int radius, int max_nn)
{
	int block_x = (points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_num;
	int grid_x = (points_num - 1) / block_x + 1;

	int *max_vid_x, *max_vid_y, *max_vid_z;
	int *min_vid_x, *min_vid_y, *min_vid_z;
	int *found_voxel_num;
	int *valid_point_tmp;

	checkCudaErrors(cudaMalloc(&max_vid_x, sizeof(int) * points_num));
	checkCudaErrors(cudaMalloc(&max_vid_y, sizeof(int) * points_num));
	checkCudaErrors(cudaMalloc(&max_vid_z, sizeof(int) * points_num));

	checkCudaErrors(cudaMalloc(&min_vid_x, sizeof(int) * points_num));
	checkCudaErrors(cudaMalloc(&min_vid_y, sizeof(int) * points_num));
	checkCudaErrors(cudaMalloc(&min_vid_z, sizeof(int) * points_num));

	chechCudaErrors(cudaMalloc(&found_voxel_num, sizeof(int) * (points_num + 1)));
	checkCudaErrors(cudaMalloc(&valid_point_tmp, sizeof(int) * (points_num + 1)));

	radiusSearch1<<<grid_x, block_x>>>(qx, qy, qz, radius, max_nn, points_num,
										global_voxel_, vgrid_x_, vgrid_y_, vgrid_z_,
										voxel_x_, voxel_y_, voxel_z_,
										min_b_x_, min_b_y_, min_b_z_,
										max_b_x_, max_b_y_, max_b_z_,
										max_vid_x, max_vid_y, max_vid_z,
										min_vid_x, min_vid_y, min_vid_z,
										valid_point_tmp, found_voxel_num);
	checkCudaErrors(cudaDeviceSynchronize());

	int *query_status;

	checkCudaErrors(cudaMalloc(&query_status, sizeof(int) * points_num));
	checkCudaErrors(cudaMemcpy(query_status, valid_point_tmp, sizeof(int) * points_num, cudaMemcpyDeviceToDevice));

	ExclusiveScan(found_voxel_num, points_num + 1, &qresult_size_);
	starting_voxel_id_ = found_voxel_num;
	ExclusiveScan(valid_point_tmp, points_num + 1, &valid_points_num_);

	checkCudaErrors(cudaMalloc(&valid_points_, sizeof(int) * valid_points_num_));
	checkCudaErrors(cudaMalloc(&voxel_id_, sizeof(int) * qresult_size_));

	collectValidPoints<<<grid_x, block_x>>>(query_status, valid_points_, valid_point_tmp, points_num);
	radiusSearch2<<<grid_x, block_x>>>(qx, qy, qz, radius, max_nn, points_num,
										global_voxel_, vgrid_x_, vgrid_y_, vgrid_z_,
										max_vid_x, max_vid_y, max_vid_z,
										min_vid_x, min_vid_y, min_vid_z,
										starting_voxel_id_, voxel_id_);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(max_vid_x));
	checkCudaErrors(cudaFree(max_vid_y));
	checkCudaErrors(cudaFree(max_vid_z));

	checkCudaErrors(cudaFree(min_vid_x));
	checkCudaErrors(cudaFree(min_vid_y));
	checkCudaErrors(cudaFree(min_vid_z));

	checkCudaErrors(cudaFree(valid_point_tmp));
}


}
