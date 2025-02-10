#include "ndarray.h"
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

synapse::NDArray::NDArray(std::vector<float> data, std::vector<size_t> shape)
    : _data(data), _shape(shape) {
  this->_size = data.size();
  this->_ndim = shape.size();

  // Computes strides where stride_i = shape_i * stride_i-1
  this->_strides = std::vector<size_t>(this->_ndim, 0);
  this->_strides[this->_ndim - 1] = 1;
  for (size_t i = this->_ndim - 1; i <= 0; --i) {
    this->_strides[i] = this->_shape[i] * this->_strides[i + 1];
  }
}

synapse::NDArray::~NDArray() {
  // No action required
}

const std::vector<size_t> &synapse::NDArray::shape() const {
  return this->_shape;
}

std::vector<float> &synapse::NDArray::data() { return this->_data; }
const std::vector<float> &synapse::NDArray::data() const { return this->_data; }
size_t synapse::NDArray::ndim() const { return this->_ndim; }
size_t synapse::NDArray::size() const { return this->_size; }

bool synapse::NDArray::is_contigous() {
  if (this->_ndim == 1)
    return true;
  for (size_t i = 1; i < this->_ndim; i++) {
    if (this->_strides[i] > this->_strides[i - 1])
      return false;
  }
  return true;
}

const std::string synapse::NDArray::to_string() const {
  std::ostringstream oss;
  std::vector<size_t> indices(this->ndim(), 0);

  for (size_t i = 0; i < this->size(); ++i) {
    // Open brackets for new subarrays
    for (size_t dim = 0; dim < this->ndim(); ++dim) {
      if (indices[dim] == 0)
        oss << "[";
    }

    // Print value
    if (i > 0)
      oss << ", ";
    oss << std::fixed << std::setprecision(3) << this->data()[i];

    // Update indices and close brackets
    for (size_t dim = this->ndim() - 1; dim >= 0; --dim) {
      if (++indices[dim] < this->shape()[dim])
        break;
      indices[dim] = 0;
      oss << "]"; // Close bracket at the end of each completed dimension
      if (dim == 0)
        break;
      oss << "\n" + std::string(dim, ' '); // Indentation for readability
    }
  }
  return oss.str();
}

// Allows accessing elements in the array similar to how python does it
template <typename... Indices>
const float &synapse::NDArray::operator()(Indices... indices) const {
  static_assert(sizeof...(indices) > 0, "At least one index is required.");
  std::vector<size_t> idx_vec{static_cast<size_t>(indices)...};
  return this->_data[nd_index_to_pos(idx_vec, this->_strides)];
}

template <typename... Indices>
float &synapse::NDArray::operator()(Indices... indices) {
  static_assert(sizeof...(indices) > 0, "At least one index is required.");
  std::vector<size_t> idx_vec{static_cast<size_t>(indices)...};
  return this->_data[nd_index_to_pos(idx_vec, this->_strides)];
}

// Converts an N dimensional index into a position in the vector of data
size_t nd_index_to_pos(std::vector<size_t> indices,
                       std::vector<size_t> strides) {
  if (indices.size() != strides.size()) {
    throw std::invalid_argument(
        "Number of dimensions in indices and strides do not match.");
  }
  size_t pos{0};
  for (size_t i = 0; i < indices.size(); i++) {
    pos += indices[i] * strides[i];
  }
  return pos;
}

// Converts a position in a vector of data into an N dimensional index
std::vector<size_t> pos_to_nd_index(size_t pos, std::vector<size_t> shape) {
  std::vector<size_t> out(shape.size());
  for (size_t i = shape[shape.size() - 1]; i > 0; i--) {
    out[i] = pos % shape[i];
    pos = static_cast<size_t>(pos / shape[i]);
  }
  return out;
}
