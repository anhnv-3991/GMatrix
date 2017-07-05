#include "Matrix.h"
#include "debug.h"

namespace gpu {

extern "C" __global__ void compareNotEqual(Matrix left, Matrix right, bool *res)
{
	for (int i = 0; i < left.getRowsCount(); i++) {
		for (int j = 0; j < left.getColsCount(); j++) {
			if (left(i, j) != right(i, j)) {
				*res = false;
				return;
			}
		}
	}

	*res = true;
}

}
