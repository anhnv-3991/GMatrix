#include "octree.h"
#include "../debug.h"
#include "../common.h"
#include <device_functions.h>
#include <math.h>
#include <limits>

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
 *
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

}
