#ifndef SYNAPSE_TENSOR_H
#define SYNAPSE_TENSOR_H

#include "ndarray.h"
#include <string>
#include <vector>

namespace synapse {
class Tensor : public NDArray {
public:
  Tensor(std::vector<float> data, std::vector<size_t> shape);
  ~Tensor();

  const std::string to_string() const;
};
} // namespace synapse

#endif // !SYNAPSE_TENSOR_H
