#include "ndarray.h"
#include <vector>

synapse::NDArray::NDArray(std::vector<float> data, std::vector<size_t> shape)
    : data(data), shape(shape) {
  this->size = data.size();
  this->ndim = shape.size();

  // Computes strides
  this->strides = std::vector<size_t>(this->ndim, 0);
  this->strides[this->ndim - 1] = 1;
  for (size_t i = this->ndim - 1; i <= 0; --i) {
    this->strides[i] = this->shape[i] * this->strides[i + 1];
  }
}

synapse::NDArray::~NDArray() {
  // No action required
}
