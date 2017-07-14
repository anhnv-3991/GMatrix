#ifndef GNDT_H_
#define GNDT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.h"
#include "MatrixHost.h"
#include "MatrixDevice.h"
#include "common.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

namespace gpu {
class GRegistration {
public:
	GRegistration();

	void align(Eigen::Matrix<float, 4, 4> &guess);

	inline void setTransformationEpsilon(double trans_eps)
	{
		transformation_epsilon_ = trans_eps;
	}

	inline void setMaximumIterations(int max_itr)
	{
		max_iterations_ = max_itr;
	}

	void setInputSource(float *x, float *y, float *z, int points_num);


	inline Eigen::Matrix<float, 4, 4> getFinalTransformation()
	{
		return final_transformation_;
	}

	virtual ~GRegistration() = 0;
protected:
	bool initCompute();
	virtual void computeTransformation(Eigen::Matrix<float, 4, 4> &guess) = 0;

	double transformation_epsilon_;
	int max_iterations_;
	float *x_, *y_, *z_;
	int points_number_;
	float *trans_x_, *trans_y_, *trans_z_;

	bool converged_;
	int nr_iterations_;

	Eigen::Matrix<float, 4, 4> final_transformation_, transformation_, previous_transformation_;

	GVoxelGrid voxel_grid_;

	bool target_cloud_updated_;
};
}

#endif
