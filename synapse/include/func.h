#ifndef FUNC_H
#define FUNC_H

#include "tensor.h"

namespace synapse {
Tensor add(Tensor &t1, Tensor &t2);
Tensor mul(Tensor &t1, Tensor &t2);

bool is_close(Tensor &t1, Tensor &t2, float tol = 1e-5f);
} // namespace synapse

#endif // !FUNC
