#include "octree.h"
#include "../debug.h"
#include "../common.h"
#include <math.h>
#include <limits>
#include <eigen3/Eigen/Eigenvalues>

namespace gpu {

extern "C" __global__ void initLeafGrid(GOctreeNode *voxel_grid, int vgrid_x, int vgrid_y, int vgrid_z, 
										int vgrid_parent_x, int vgrid_parent_y, int vgrid_parent_z, int parent_head,
										float min_xy, float min_yz, float min_zx,
										float voxel_x, float voxel_y, float voxel_z,
										float *centroid_buff, float *cov_buff)
{
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_z = threadIdx.z + blockIdx.z * blockDim.z;

	if (id_x < vgrid_x && id_y < vgrid_y && id_z < vgrid_z) {
		int vgrid_id = id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y;
		int pid_x = ((id_x / 2) < vgrid_parent_x) ? id_x / 2 : -1;
		int pid_y = ((id_y / 2) < vgrid_parent_y) ? id_y / 2 : -1;
		int pid_z = ((id_z / 2) < vgrid_parent_z) ? id_z / 2 : -1;
		int cid_x = id_x * 2;
		int cid_y = id_y * 2;
		int cid_z = id_z * 2;

		GOctreeNode *voxel = voxel_grid + vgrid_id;

		voxel->parent() = parent_head + pid_x + pid_y * vgrid_x + pid_z * vgrid_x * vgrid_y;

		voxel->offset() = 0;
		voxel->setChildBuf(NULL);
		
		voxel->minXY() = min_xy + id_z * voxel_z;
		voxel->minYZ() = min_yz + id_x * voxel_x;
		voxel->minZX() = min_zx + id_y * voxel_y;

		voxel->maxXY() = min_xy + id_z * voxel_z + voxel_z;
		voxel->maxYZ() = min_yz + id_x * voxel_x + voxel_x;
		voxel->maxZX() = min_zx + id_y * voxel_y + voxel_y;

		voxel->centroid() = MatrixDevice(1, 3, vgrid_x * vgrid_y * vgrid_z, centroid_buff);
		voxel->covariance() = MatrixDevice(3, 3, vgrid_x * vgrid_y * vgrid_z, cov_buff);

		voxel->isLeaf() = true;
		voxel->pointNum() = 0;
		voxel->level() = 1;
	}
}
extern "C" __global__ void initNonLeafGrid(GOctreeNode *voxel_grid, int vgrid_x, int vgrid_y, int vgrid_z, int current_head, int *child,
											int vgrid_parent_x, int vgrid_parent_y, int vgrid_parent_z, int parent_head,
											int vgrid_child_x, int vgrid_child_y, int vgrid_child_z, int child_head, 
											float min_xy, float min_yz, float min_zx, 
											float voxel_x, float voxel_y, float voxel_z, int level)
{
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_z = threadIdx.z + blockIdx.z * blockDim.z;

	if (id_x < vgrid_x && id_y < vgrid_y && id_z < vgrid_z) {
		int vgrid_id = id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y;
		int pid_x = ((id_x / 2) < vgrid_parent_x) ? id_x / 2 : -1;
		int pid_y = ((id_y / 2) < vgrid_parent_y) ? id_y / 2 : -1;
		int pid_z = ((id_z / 2) < vgrid_parent_z) ? id_z / 2 : -1;
		int cid_x = id_x * 2;
		int cid_y = id_y * 2;
		int cid_z = id_z * 2;

		GOctreeNode *voxel = voxel_grid + vgrid_id;

		voxel->parent() = parent_head + pid_x + pid_y * vgrid_x + pid_z * vgrid_x * vgrid_y;

		voxel->offset() = vgrid_x * vgrid_y * vgrid_z;
		voxel->setChildBuf(child + current_head + vgrid_id);

		voxel->child(0) = (cid_x >= vgrid_child_x && cid_y >= vgrid_child_y && cid_z >= vgrid_child_z) ? -1 : vgrid_id * 2 + child_head;
		voxel->child(1) = (cid_x >= vgrid_child_x && cid_y >= vgrid_child_y && cid_z + 1 >= vgrid_child_z) ? -1 : vgrid_id * 2 + 1 + child_head;
		voxel->child(2) = (cid_x >= vgrid_child_x && cid_y + 1 >= vgrid_child_y && cid_z >= vgrid_child_z) ? -1 : vgrid_id * 2 + 2 + child_head; 
		voxel->child(3) = (cid_x >= vgrid_child_x && cid_y + 1 >= vgrid_child_y && cid_z + 1 >= vgrid_child_z) ? -1 : vgrid_id * 2 + 3 + child_head;
		voxel->child(4) = (cid_x + 1 >= vgrid_child_x && cid_y >= vgrid_child_y && cid_z >= vgrid_child_z) ? -1 : vgrid_id * 2 + 4 + child_head;
		voxel->child(5) = (cid_x + 1 >= vgrid_child_x && cid_y >= vgrid_child_y && cid_z + 1 >= vgrid_child_z) ? -1 : vgrid_id * 2 + 5 + child_head;
		voxel->child(6) = (cid_x + 1 >= vgrid_child_x && cid_y + 1 >= vgrid_child_y && cid_z >= vgrid_child_z) ? -1 : vgrid_id * 2 + 6 + child_head;
		voxel->child(7) = (cid_x + 1 >= vgrid_child_x && cid_y + 1 >= vgrid_child_y && cid_z + 1 >= vgrid_child_z) ? -1 : vgrid_id * 2 + 7 + child_head;
		
		voxel->minXY() = min_xy + id_z * voxel_z;
		voxel->minYZ() = min_yz + id_x * voxel_x;
		voxel->minZX() = min_zx + id_y + voxel_y;

		voxel->maxXY() = min_xy + id_z * voxel_z + voxel_z;
		voxel->maxYZ() = min_yz + id_x * voxel_x + voxel_x;
		voxel->maxZX() = min_zx + id_y * voxel_y + voxel_y;

		voxel->isLeaf() = false;
		voxel->pointNum() = 0;
		voxel->level() = level;
	}
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
										GOctreeNode *voxel_grid, int vgrid_x, int vgrid_y, int vgrid_z,
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

		GOctreeNode *voxel = voxel_grid + voxel_id;

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
extern "C" __global__ void updateVoxelCentroid(GOctreeNode *voxel_grid, int vgrid_x, int vgrid_y, int vgrid_z)
{
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_z = threadIdx.z + blockIdx.z * blockDim.z;

	if (id_x < vgrid_x && id_y < vgrid_y && id_z < vgrid_z) {
		int vgrid_id = id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y;
		GOctreeNode *node = voxel_grid + vgrid_id;
		int points_num = node->pointNum();
		
		MatrixDevice centr = voxel_grid[vgrid_id].centroid();
		MatrixDevice cov = voxel_grid[vgrid_id].covariance();
		MatrixDevice icov = voxel_grid[vgrid_id].inverseCovariance();

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

//Input are supposed to be in device memory
void GOctree::setInput(float *x, float *y, float *z, int points_num)
{
	x_ = x;
	y_ = y;
	z_ = z;
	points_num_ = points_num;

	findBoundaries();

	checkCudaErrors(cudaMalloc(&global_voxel_, sizeof(GOctreeNode) * vgrid_x_ * vgrid_y_ * vgrid_z_ * 2));



}

extern "C" __global__ void findMax(float *x, float *y, float *z, int in_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ float s_x[BLOCK_SIZE_X], s_y[BLOCK_SIZE_X], s_z[BLOCK_SIZE_X];
	float tmp_x, tmp_y, tmp_z;

	if (index < in_size / 2) {
		tmp_x = x[index];
		tmp_y = y[index];
		tmp_z = z[index];
		s_x[threadIdx.x] = (index + stride < in_size) ? ((tmp_x < x[index + stride]) ? x[index + stride] : tmp_x) : tmp_x;
		s_y[threadIdx.x] = (index + stride < in_size) ? ((tmp_y < x[index + stride]) ? x[index + stride] : tmp_y) : tmp_y;
		s_z[threadIdx.x] = (index + stride < in_size) ? ((tmp_z < x[index + stride]) ? x[index + stride] : tmp_z) : tmp_z;

		__syncthreads();

		stride = (blockIdx.x * blockDim.x + blockDim.x <= in_size) ? blockDim.x : in_size - blockIdx.x * blockDim.x;

		stride /= 2;

		while (stride > 0 && threadIdx.x < stride) {
			s_x[threadIdx.x] = (s_x[threadIdx.x] > s_x[threadIdx.x + stride]) ? s_x[threadIdx.x] : s_x[threadIdx.x + stride];
			__syncthreads();
		}
	}
}

extern "C" __global__ void findMin(float *x, float *y, float *z, int in_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	__shared__ float s_x[BLOCK_SIZE_X], s_y[BLOCK_SIZE_X], s_z[BLOCK_SIZE_X];
	float tmp_x, tmp_y, tmp_z;

	if (index < in_size / 2) {
		tmp_x = x[index];
		tmp_y = y[index];
		tmp_z = z[index];
		s_x[threadIdx.x] = (index + stride < in_size) ? ((tmp_x > x[index + stride]) ? x[index + stride] : tmp_x) : tmp_x;
		s_y[threadIdx.x] = (index + stride < in_size) ? ((tmp_y > x[index + stride]) ? x[index + stride] : tmp_y) : tmp_y;
		s_z[threadIdx.x] = (index + stride < in_size) ? ((tmp_z > x[index + stride]) ? x[index + stride] : tmp_z) : tmp_z;

		__syncthreads();

		stride = (blockIdx.x * blockDim.x + blockDim.x <= in_size) ? blockDim.x : in_size - blockIdx.x * blockDim.x;

		stride /= 2;

		while (stride > 0 && threadIdx.x < stride) {
			s_x[threadIdx.x] = (s_x[threadIdx.x] > s_x[threadIdx.x + stride]) ? s_x[threadIdx.x] : s_x[threadIdx.x + stride];
			__syncthreads();
		}
	}
}

void GOctree::findBoundaries()
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

	int half_points_num;

	while (points_num > 0) {
		half_points_num = points_num / 2;

		int block_x = (half_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_points_num;
		int grid_x = (half_points_num - 1) / block_x + 1;

		findMax<<<grid_x, block_x>>>(max_x, max_y, max_z, points_num);
		findMin<<<grid_x, block_x>>>(min_x, min_y, min_z, points_num);

		points_num /= 2;
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


extern "C" __global__ void radiusSearch(float *x, float *y, float *z, int radius, int max_nn, int points_num,
											GOctreeNode *grid, int vgrid_x, int vgrid_y, int vgrid_z,
											float voxel_x, float voxel_y, float voxel_z,
											int min_b_x, int min_b_y, int min_b_z,
											int max_b_x, int max_b_y, int max_b_z,
											float max_x, float max_y, float max_z,
											float min_x, float min_y, float min_z,
											MatrixDevice *point_gradients, MatrixDevice *point_hessians,
											MatrixDevice j_ang_a, MatrixDevice j_ang_b, MatrixDevice j_ang_c, MatrixDevice j_ang_d,
											MatrixDevice j_ang_e, MatrixDevice j_ang_f, MatrixDevice j_ang_g, MatrixDevice j_ang_h,
											MatrixDevice h_ang_a2_, MatrixDevice h_ang_a3_, MatrixDevice h_ang_b2_, MatrixDevice h_ang_b3_, MatrixDevice h_ang_c2_,
											MatrixDevice h_ang_c3_, MatrixDevice h_ang_d1_, MatrixDevice h_ang_d2_, MatrixDevice h_ang_d3_, MatrixDevice h_ang_e1_,
											MatrixDevice h_ang_e2_, MatrixDevice h_ang_e3_, MatrixDevice h_ang_f1_, MatrixDevice h_ang_f2_, MatrixDevice h_ang_f3_,
											float gauss_d1, float gauss_d2)
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

		MatrixDevice g = point_gradients[i];
		MatrixDevice H = point_hessians[i];
		int nn = 0;

		for (int j = min_id_x; j <= max_id_x && nn < max_nn; j++) {
			for (int k = min_id_y; k <= max_id_y && nn < max_nn; k++) {
				for (int l = min_id_z; l <= max_id_z && nn < max_nn; l++) {
					int voxel_id = j + k * vgrid_x + l * vgrid_x * vgrid_y;
					GOctreeNode *voxel = grid[voxel_id];

					if (voxel->pointNum() > 0) {
						MatrixDevice centroid = voxel->centroid();

						if (squareDistance(centroid(0), centroid(1), centroid(2), t_x, t_y, t_z) <= radius * radius) {
							nn++;

							g(1, 3) = t_x * j_ang_a(0) + t_y * j_ang_a(1) + t_z * t_ang_a(2);
							g(2, 3) = t_x * j_ang_b(0) + t_y * j_ang_b(1) + t_z * t_ang_b(2);
							g(0, 4) = t_x * j_ang_c(0) + t_y * j_ang_c(1) + t_z * j_ang_c(2);
							g(1, 4) = t_x * j_ang_d(0) + t_y * j_ang_d(1) + t_z * j_ang_d(2);
							g(2, 4) = t_x * j_ang_e(0) + t_y * j_ang_e(1) + t_z * j_ang_e(2);
							g(0, 5) = t_x * j_ang_f(0) + t_y * j_ang_f(1) + t_z * j_ang_f(2);
							g(1, 5) = t_x * j_ang_g(0) + t_y * j_ang_g(1) + t_z * j_ang_g(2);
							g(2, 5) = t_x * j_ang_h(0) + t_y * j_ang_h(1) + t_z * j_ang_h(2);

							H(9, 3) = 0;
							H(10, 3) = t_x * h_ang_a2(0) + t_y * h_ang_a2(1) + t_z * h_ang_a2(2);
							H(11, 3) = t_x * h_ang_a3(0) + t_y * h_ang_a3(1) + t_z * h_ang_a3(2);

							H(12, 3) = H(9, 4) = 0;
							H(13, 3) = H(10, 4) = t_x * h_ang_b2(0) + t_y * h_ang_b2(1) + t_z * h_ang_b2(2);
							H(14, 3) = H(11, 4) = t_x * h_ang_b3(0) + t_y * h_ang_b3(1) + t_z * h_ang_b3(2);

							H(15, 3) = 0;
							H(16, 3) = H(9, 5) = t_x * h_ang_c2(0) + t_y * h_ang_c2(1) + t_z * h_ang_c2(2);
							H(17, 3) = H(10, 5) = t_x * h_ang_c3(0) + t_y * h_ang_c3(1) + t_z * h_ang_c3(2);

							H(12, 4) = t_x * h_ang_d1(0) + t_y * h_ang_d1(1) + t_z * h_ang_d1(2);
							H(13, 4) = t_x * h_ang_d2(0) + t_y * h_ang_d2(1) + t_z * h_ang_d2(2);
							H(14, 4) = t_x * h_ang_d3(0) + t_y * h_ang_d3(1) + t_z * h_ang_d3(2);

							H(15, 4) = H(12, 5) = t_x * h_ang_e1(0) + t_y * h_ang_e1(1) + t_z * h_ang_e1(2);
							H(16, 4) = H(13, 5) = t_x * h_ang_e2(0) + t_y * h_ang_e2(1) + t_z * h_ang_e2(2);
							H(17, 4) = H(14, 5) = t_x * h_ang_e3(0) + t_y * h_ang_e3(1) + t_z * h_ang_e3(2);

							H(15, 5) = t_x * h_ang_f1(0) + t_y * h_ang_f1(1) + t_z * h_ang_f1(2);
							H(16, 5) = t_x * h_ang_f2(0) + t_y * h_ang_f2(1) + t_z * h_ang_f2(2);
							H(17, 5) = t_x * h_ang_f3(0) + t_y * h_ang_f3(1) + t_z * h_ang_f3(2);

							MatrixDevice icov = voxel->inverseCovariance();

							float cov_dxd_pi_x, cov_dxd_pi_y, cov_dxd_pi_z;

							t_x -= centroid(0);
							t_y -= centroid(1);
							t_z -= centroid(2);

							double e_x_cov_x = expf(-gauss_d2 * ((t_x * icov(0, 0) + t_y * icov(1, 0) + t_z * icov(2, 0)) * t_x
																+ ((t_x * icov(0, 1) + t_y * icov(1, 1) + t_z * icov(2, 1)) * t_y)
																+ ((t_x * icov(0, 2) + t_y * icov(1, 2) + t_z * icov(2, 2)) * t_z)) / 2);
							double score_inc = -gauss_d1 * e_x_cov_x;

							e_x_cov_x *= gauss_d2;

							e_x_cov_x *= gauss_d1;




							  Eigen::Vector3d cov_dxd_pi;
							  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
							  double e_x_cov_x = exp (-gauss_d2_ * x_trans.dot (c_inv * x_trans) / 2);
							  // Calculate probability of transtormed points existance, Equation 6.9 [Magnusson 2009]
							  double score_inc = -gauss_d1_ * e_x_cov_x;

							  e_x_cov_x = gauss_d2_ * e_x_cov_x;

							  // Error checking for invalid values.
							  if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x)
							    return (0);

							  // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
							  e_x_cov_x *= gauss_d1_;


							  for (int i = 0; i < 6; i++)
							  {
							    // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
							    cov_dxd_pi = c_inv * point_gradient_.col (i);

							    // Update gradient, Equation 6.12 [Magnusson 2009]
							    score_gradient (i) += x_trans.dot (cov_dxd_pi) * e_x_cov_x;

							    if (compute_hessian)
							    {
							      for (int j = 0; j < hessian.cols (); j++)
							      {
							        // Update hessian, Equation 6.13 [Magnusson 2009]
							        hessian (i, j) += e_x_cov_x * (-gauss_d2_ * x_trans.dot (cov_dxd_pi) * x_trans.dot (c_inv * point_gradient_.col (j)) +
							                                    x_trans.dot (c_inv * point_hessian_.block<3, 1>(3 * i, j)) +
							                                    point_gradient_.col (j).dot (cov_dxd_pi) );
							      }
							    }
							  }

							  return (score_inc);
						}
					}
				}
			}
		}

		int voxel_id = voxelId(t_x, t_y, t_z, voxel_x, voxel_y, voxel_z, min_b_x, min_b_y, min_b_z, vgrid_x, vgrid_y, vgrid_z);



		int id_x = static_cast<int>(floor(x / voxel_x) - static_cast<float>(min_b_x));
		int id_y = static_cast<int>(floor(y / voxel_y) - static_cast<float>(min_b_y));
		int id_z = static_cast<int>(floor(z / voxel_z) - static_cast<float>(min_b_z));

		return (id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y);

		int
	}

}
