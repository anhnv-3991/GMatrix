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
	double computeDerivatives(MatrixDevice score_gradient, MatrixDevice hessian,
								float *source_x, float *source_y, float *source_z,
								MatrixDevice pose, bool compute_hessian);
	void computeAngleDerivatives(MatrixHost pose, bool compute_hessian);

private:
	void transformPointCloud(float *in_x, float *in_y, float *in_z,
								float *out_x, float *out_y, float *out_z,
								int points_number, const Matrix transform);

	double gauss_d1_, gauss_d2_;
	double outlier_ratio_;
	MatrixHost j_ang_a_, j_ang_b_, j_ang_c_, j_ang_d_, j_ang_e_, j_ang_f_, j_ang_g_, j_ang_h_;

	MatrixHost h_ang_a2_, h_ang_a3_, h_ang_b2_, h_ang_b3_, h_ang_c2_, h_ang_c3_, h_ang_d1_, h_ang_d2_, h_ang_d3_,
				h_ang_e1_, h_ang_e2_, h_ang_e3_, h_ang_f1_, h_ang_f2_, h_ang_f3_;
};
}

#endif
