#ifndef GPU_OCTREE_H_
#define GPU_OCTREE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../common.h"
#include "../MatrixDevice.h"
#include <limits>

namespace gpu {
class GOctreeNode {
public:
	CUDAH GOctreeNode() :
			parent_(-1),
			child_(NULL),
			offset_(0),
			max_xy_(0),
			max_yz_(0),
			max_zx_(0),
			min_xy_(0),
			min_yz_(0),
			min_zx_(0),
			is_leaf_(false),
			point_num_(0),
			level_(-1){};

	CUDAH GOctreeNode(int parent,
						int *child,
						int offset,
						float max_xy, float max_yz, float max_zx,
						float min_xy, float min_yz, float min_zx,
						bool is_leaf,
						int point_num,
						int level):
			parent_(parent),
			child_(child),
			offset_(offset),
			max_xy_(max_xy),
			max_yz_(max_yz),
			max_zx_(max_zx),
			min_xy_(min_xy),
			min_yz_(min_yz),
			min_zx_(min_zx),
			is_leaf_(is_leaf),
			point_num_(point_num),
			level_(level) {};

	CUDAH int& parent() { return parent_; }
	CUDAH int& offset() { return offset_; }
	CUDAH int& child(int child_idx) { return child_[child_idx * offset_]; }

	CUDAH MatrixDevice& centroid() { return centroid_; }

	CUDAH MatrixDevice& covariance() { return cov_; }
	
	CUDAH MatrixDevice& inverseCovariance() { return icov_; }

	CUDAH float sizeXY() { return max_xy_ - min_xy_; }
	CUDAH float sizeYZ() { return max_yz_ - min_yz_; }
	CUDAH float sizeZX() { return max_zx_ - min_zx_; }
	CUDAH bool& isLeaf() { return is_leaf_; }
	
	CUDAH int& pointNum() { return point_num_; }
	CUDAH int* pointNumAddress() { return &point_num_; }
	CUDAH int& level() { return level_; }

	CUDAH float& maxXY() { return max_xy_; }
	CUDAH float& maxYZ() { return max_yz_; }
	CUDAH float& maxZX() { return max_zx_; }

	CUDAH float& minXY() { return min_xy_; }
	CUDAH float& minYZ() { return min_yz_; }
	CUDAH float& minZX() { return min_zx_; }

	CUDAH void setChildBuf(int *child) { child_ = child; }
	CUDAH int* getChildBuf() { return child_; }
	
protected:
	int parent_;
	int *child_;
	int offset_;
	float max_xy_, max_yz_, max_zx_;
	float min_xy_, min_yz_, min_zx_;
	bool is_leaf_;
	int point_num_;
	int level_;
	MatrixDevice cov_;
	MatrixDevice centroid_;
	MatrixDevice icov_;
};

class GOctree {
public:
	GOctree(): x_(NULL), y_(NULL), z_(NULL), points_num_(0), global_voxel_(NULL), voxel_num_(0) {} ;

	void setInput(float *x, float *y, float *z, int points_num);

	void setMinVoxelSize(int size);

	CUDAH void radiusSearch(float qx, float qy, float qz, int radius);

private:

	void build();

	void findBoundaries();

	//Coordinate of input points
	float *x_, *y_, *z_;
	int points_num_;
	GOctreeNode *global_voxel_;

	int voxel_num_;
	float max_x_, max_y_, max_z_;
	float min_x_, min_y_, min_z_;
	float voxel_x_, voxel_y_, voxel_z_;

	int max_b_x_, max_b_y_, max_b_z_;
	int min_b_x_, min_b_y_, min_b_z_;
	int vgrid_x_, vgrid_y_, vgrid_z_;
};
}

#endif
