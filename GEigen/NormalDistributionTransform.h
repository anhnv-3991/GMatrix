#ifndef GPU_NDT_H_
#define GPU_NDT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"

namespace gpu {
class GNormalDistributionTransform: protected GRegistration {
public:
protected:
	void computeTransform();

private:
	double gauss_d1_, gauss_d2_;
	double outlier_ratio_;
};
}

#endif
