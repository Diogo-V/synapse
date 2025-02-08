#ifndef SYNAPSE_TENSOR_H
#define SYNAPSE_TENSOR_H

#include "ndarray.h"
#include <vector>

namespace synapse {
class Tensor {
public:
  NDArray data;
  Tensor(std::vector<float> data, std::vector<size_t> shape);
  ~Tensor();
};
} // namespace synapse

#endif // !SYNAPSE_TENSOR_H
