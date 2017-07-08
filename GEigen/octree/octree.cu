#include "octree.h"
#include "../debug.h"
#include "../common.h"

namespace gpu {

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

extern "C" __global__ void initVoxelGrid(GOctreeNode *voxel_grid, int vgrid_x, int vgrid_y, int vgrid_z)
{
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_z = threadIdx.z + blockIdx.z * blockDim.z;

	if (id_x < vgrid_x && id_y < vgrid_y && id_z < vgrid_z) {
		int vgrid_id = id_x + id_y * vgrid_x + id_z * vgrid_x * vgrid_y;
		voxel_grid[vgrid_id].parent() = -1;

		voxel_grid[vgrid_id].child(0) = -1;
		voxel_grid[vgrid_id].child(1) = -1;
		voxel_grid[vgrid_id].child(2) = -1;
		voxel_grid[vgrid_id].child(3) = -1;
		voxel_grid[vgrid_id].child(4) = -1;
		voxel_grid[vgrid_id].child(5) = -1;
		voxel_grid[vgrid_id].child(6) = -1;
		voxel_grid[vgrid_id].child(7) = -1;

		voxel_grid[vgrid_id].offset() =
	}


	int parent_;
	int *child_;
	int offset_;
	float centroid_x_, centroid_y_, centroid_z_;
	float size_x_, size_y_, size_z_;
	bool is_leaf_;
	int point_num_;
	int level_;
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
	__shared__ float s_x[BLOCK_SIZE_X], s_y[BLOCK_SIZE_Y], s_z[BLOCK_SIZE_X];
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
