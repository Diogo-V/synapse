#include "func.h"
#include "tensor.h"
#include <stdexcept>
#include <vector>

synapse::Tensor synapse::add(const synapse::Tensor &t1,
                             const synapse::Tensor &t2) {
  if (t1.data.size() != t2.data.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  Tensor t3{std::vector<float>(t1.data.size()), t1.data.shape()};
  for (size_t i = 0; i < t1.data.size(); i++) {
    t3.data.data()[i] = t1.data.data()[i] + t2.data.data()[i];
  }
  return t3;
}

synapse::Tensor synapse::mul(const synapse::Tensor &t1,
                             const synapse::Tensor &t2) {
  if (t1.data.size() != t2.data.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  Tensor t3{std::vector<float>(t1.data.size()), t1.data.shape()};
  for (size_t i = 0; i < t1.data.size(); i++) {
    t3.data.data()[i] = t1.data.data()[i] * t2.data.data()[i];
  }
  return t3;
}

synapse::Tensor synapse::matmul(const synapse::Tensor &t1,
                                const synapse::Tensor &t2) {
  if (t1.data.size() != t2.data.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  // TODO(diogo): I need to implement shape broadcast

  return t1;
}

bool synapse::is_close(const synapse::Tensor &t1, const synapse::Tensor &t2,
                       double tol) {
  if (t1.data.shape() != t2.data.shape())
    return false;
  for (size_t i = 0; i < t1.data.size(); ++i) {
    if (std::fabs(t1.data.data()[i] - t2.data.data()[i]) > tol)
      return false;
  }
  return true;
}
