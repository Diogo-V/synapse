#include "tensor.h"
#include "ndarray.h"
#include <vector>

synapse::Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape)
    : data(synapse::NDArray(data, shape)) {
  // Does not need to initialize anything else
}

synapse::Tensor::~Tensor() {
  // Does not need to clean anything
}
