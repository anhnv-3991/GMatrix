#ifndef GPU_OCTREE_H_
#define GPU_OCTREE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../common.h"
#include "../MatrixDevice.h"
#include <float.h>

namespace gpu {
class GVoxel {
public:
	CUDAH GVoxel() :
			max_xy_(0),
			max_yz_(0),
			max_zx_(0),
			min_xy_(0),
			min_yz_(0),
			min_zx_(0),
			point_num_(0){};

	CUDAH GVoxel(float max_xy, float max_yz, float max_zx,
					float min_xy, float min_yz, float min_zx,
					int point_num):
			max_xy_(max_xy),
			max_yz_(max_yz),
			max_zx_(max_zx),
			min_xy_(min_xy),
			min_yz_(min_yz),
			min_zx_(min_zx),
			point_num_(point_num){};

	CUDAH MatrixDevice& centroid() { return centroid_; }

	CUDAH MatrixDevice& covariance() { return cov_; }
	
	CUDAH MatrixDevice& inverseCovariance() { return icov_; }

	CUDAH float sizeXY() { return max_xy_ - min_xy_; }
	CUDAH float sizeYZ() { return max_yz_ - min_yz_; }
	CUDAH float sizeZX() { return max_zx_ - min_zx_; }
	
	CUDAH int& pointNum() { return point_num_; }
	CUDAH int* pointNumAddress() { return &point_num_; }

	CUDAH float& maxXY() { return max_xy_; }
	CUDAH float& maxYZ() { return max_yz_; }
	CUDAH float& maxZX() { return max_zx_; }

	CUDAH float& minXY() { return min_xy_; }
	CUDAH float& minYZ() { return min_yz_; }
	CUDAH float& minZX() { return min_zx_; }
	
protected:
	float max_xy_, max_yz_, max_zx_;
	float min_xy_, min_yz_, min_zx_;
	int point_num_;
	MatrixDevice cov_;
	MatrixDevice centroid_;
	MatrixDevice icov_;
};

class GVoxelGrid {
public:
	GVoxelGrid():
		x_(NULL),
		y_(NULL),
		z_(NULL),
		points_num_(0),
		global_voxel_(NULL),
		voxel_num_(0),
		max_x_(FLT_MAX),
		max_y_(FLT_MAX),
		max_z_(FLT_MAX),
		min_x_(FLT_MIN),
		min_y_(FLT_MIN),
		min_z_(FLT_MIN),
		voxel_x_(0),
		voxel_y_(0),
		voxel_z_(0),
		max_b_x_(0),
		max_b_y_(0),
		max_b_z_(0),
		min_b_x_(0),
		min_b_y_(0),
		min_b_z_(0),
		vgrid_x_(0),
		vgrid_y_(0),
		vgrid_z_(0) {};

	void setInput(float *x, float *y, float *z, int points_num);

	void setMinVoxelSize(int size);

	void radiusSearch(float *qx, float *qy, float *qz, int points_num, int radius, int max_nn);

	int *getValidPoints() {
		return valid_points_;
	}

	int *getNeighborIds() {
		return neighbor_id_;
	}

	int *getVoxelIds() {
		return voxel_id_;
	}

	int getSearchResultSize() {
		return qresult_size_;
	}

	int getValidPointsNum() {
		return valid_points_num_;
	}

	GVoxel *getVoxelList() {
		return global_voxel_;
	}

	int getVoxelX() { return voxel_x_; }
	int getVoxelY() { return voxel_y_; }
	int getVoxelZ() { return voxel_z_; }

	int getVoxelNum() { return voxel_num_; }

	float getMaxX() { return max_x_; }
	float getMaxY() { return max_y_; }
	float getMaxZ() { return max_z_; }

	float getMinX() { return min_x_; }
	float getMinY() { return min_y_; }
	float getMinZ() { return min_z_; }

	float getVoxelX() { return voxel_x_; }
	float getVoxelY() { return voxel_y_; }
	float getVoxelZ() { return voxel_z_; }

	int getMaxBX() { return max_b_x_; }
	int getMaxBY() { return max_b_y_; }
	int getMaxBZ() { return max_b_z_; }

	int getMinBX() { return min_b_x_; }
	int getMinBY() { return min_b_y_; }
	int getMinBZ() { return min_b_z_; }

	int getVgridX() { return vgrid_x_; }
	int getVgridY() { return vgrid_y_; }
	int getVgridZ() { return vgrid_z_; }

private:

	void initialize();

	void insertPoints();

	void findBoundaries();

	template <typename T = int>
	void GVoxelGrid::ExclusiveScan(T *input, int ele_num, T *sum);

	//Coordinate of input points
	float *x_, *y_, *z_;
	int points_num_;
	GVoxel *global_voxel_;

	int voxel_num_;
	float max_x_, max_y_, max_z_;
	float min_x_, min_y_, min_z_;
	float voxel_x_, voxel_y_, voxel_z_;

	int max_b_x_, max_b_y_, max_b_z_;
	int min_b_x_, min_b_y_, min_b_z_;
	int vgrid_x_, vgrid_y_, vgrid_z_;

	//Array storing results of radius search
	int *valid_points_, *starting_voxel_id_, *voxel_id_;
	int qresult_size_, valid_points_num_;
};
}

#endif
