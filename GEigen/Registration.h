#ifndef GNDT_H_
#define GNDT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.h"
#include "SquareMatrix.h"
#include "Vector.h"
#include "common.h"

namespace gpu {
class GRegistration {
public:
	GRegistration();

	void align();

	void setTransformationEpsilon(double trans_eps);

	void setStepSize(double step_size);

	void setResolution(float resolution);

	void setMaximumIterations(int max_itr);

	void setInputSource(float *x, float *y, float *z, int points_num);

	void setInitGuess(const Matrix input);

	virtual ~GRegistration() = 0;
protected:
	virtual void computeTransformation() = 0;

	double trans_epsilon_;
	double step_size_;
	float resolution_;
	int max_iterations_;
	float *x_, *y_, *z_;
	int points_number_;
	Matrix init_guess_;
	float *out_x_, *out_y_, *out_z_;
	int out_points_num_;

	bool converged_;
	Matrix transformation_, final_transformation_, previous_transformation_;
	int nr_iterations_;

	Matrix point_gradient_, point_hessian_;

};
}

#endif
