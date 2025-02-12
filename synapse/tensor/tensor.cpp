#include "tensor.h"
#include "ndarray.h"
#include <string>
#include <vector>

synapse::Tensor::Tensor(std::vector<float> data, synapse::Shape shape)
    : synapse::NDArray(data, shape) {
  // Does not need to initialize anything else
}

synapse::Tensor::~Tensor() {
  // Does not need to clean anything
}

const std::string synapse::Tensor::to_string() const {
  std::string out{""};
  out += synapse::NDArray::to_string();
  return out;
}
