#ifndef FUNC_H
#define FUNC_H

#include "tensor.h"

namespace synapse {
Tensor add(const Tensor &t1, const Tensor &t2);
Tensor mul(const Tensor &t1, const Tensor &t2);
Tensor matmul(const Tensor &t1, const Tensor &t2);

bool is_close(const Tensor &t1, const Tensor &t2, double tol = 1e-5f);
} // namespace synapse

#endif // !FUNC
