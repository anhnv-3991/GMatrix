#ifndef GPU_OCTREE_H_
#define GPU_OCTREE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "../common.h"

namespace gpu {
class GOctreeNode {
public:
	CUDAH GOctreeNode() :
			parent_(-1),
			child_(NULL),
			offset_(0),
			centroid_x_(0),
			centroid_y_(0),
			centroid_z_(0),
			size_x_(0),
			size_y_(0),
			size_z_(0),
			is_leaf_(false),
			point_num_(0),
			level_(-1) {};

	CUDAH GOctreeNode(int parent,
						int *child,
						int offset,
						float centroid_x, float centroid_y, float centroid_z,
						float size_x, float size_y, float size_z,
						bool is_leaf,
						int point_num,
						int level):
			parent_(parent),
			child_(child),
			offset_(offset),
			centroid_x_(centroid_x),
			centroid_y_(centroid_y),
			centroid_z_(centroid_z),
			size_x_(size_x),
			size_y_(size_y),
			size_z_(size_z),
			is_leaf_(is_leaf),
			point_num_(point_num),
			level_(level) {};

	CUDAH int& parent() { return parent_; }
	CUDAH int& child(int child_idx) { return child_[child_idx * offset_]; }
	CUDAH int& offset() { return offset_; }
	CUDAH float& centroidX() { return centroid_x_; }
	CUDAH float& centroidY() { return centroid_y_; }
	CUDAH float& centroidZ() { return centroid_z_; }
	CUDAH int& voxelNum() { return voxel_num_; }
	CUDAH float& sizeX() { return size_x_; }
	CUDAH float& sizeY() { return size_y_; }
	CUDAH float& sizeZ() { return size_z_; }
	CUDAH bool& isLeaf() { return is_leaf_; }
	CUDAH int& pointNum() { return point_num_; }
	CUDAH int& level() { return level_; }

private:
	int parent_;
	int *child_;
	int offset_;
	float centroid_x_, centroid_y_, centroid_z_;
	float size_x_, size_y_, size_z_;
	bool is_leaf_;
	int point_num_;
	int level_;
};

class GOctree {
public:
	GOctree(): x_(NULL), y_(NULL), z_(NULL), points_num_(0), global_voxel_(NULL), voxel_num_(0), min_voxel_size_(0), max_voxel_size_(0) {} ;

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
	float max_voxel_size_;
	float max_x_, max_y_, max_z_;
	float min_x_, min_y_, min_z_;
	float voxel_x_, voxel_y_, voxel_z_;

	int max_b_x_, max_b_y_, max_b_z_;
	int min_b_x_, min_b_y_, min_b_z_;
	int vgrid_x_, vgrid_y_, vgrid_z_;
};
}

#endif
