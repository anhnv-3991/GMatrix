#ifndef GPU_NDT_H_
#define GPU_NDT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Registration.h"
#include "common.h"

namespace gpu {
class GNormalDistributionTransform: protected GRegistration {
public:
protected:
	void computeTransformation();

private:
	void transformPointCloud(float *in_x, float *in_y, float *in_z,
								float *out_x, float *out_y, float *out_z,
								int points_number, const Matrix transform);

	double gauss_d1_, gauss_d2_;
	double outlier_ratio_;
};
}

#endif
