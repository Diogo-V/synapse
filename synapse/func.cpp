#include "func.h"
#include "tensor.h"
#include <stdexcept>
#include <vector>

synapse::Tensor synapse::add(const synapse::Tensor &t1,
                             const synapse::Tensor &t2) {
  if (t1.size() != t2.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  Tensor t3{std::vector<float>(t1.size()), t1.shape()};
  for (size_t i = 0; i < t1.size(); i++) {
    t3.data()[i] = t1.data()[i] + t2.data()[i];
  }
  return t3;
}

synapse::Tensor synapse::mul(const synapse::Tensor &t1,
                             const synapse::Tensor &t2) {
  if (t1.size() != t2.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  Tensor t3{std::vector<float>(t1.size()), t1.shape()};
  for (size_t i = 0; i < t1.size(); i++) {
    t3.data()[i] = t1.data()[i] * t2.data()[i];
  }
  return t3;
}

synapse::Tensor synapse::matmul(const synapse::Tensor &t1,
                                const synapse::Tensor &t2) {
  if (t1.size() != t2.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  // TODO(diogo): I need to implement shape broadcast

  return t1;
}

bool synapse::is_close(const synapse::Tensor &t1, const synapse::Tensor &t2,
                       double tol) {
  if (t1.shape() != t2.shape())
    return false;
  for (size_t i = 0; i < t1.size(); ++i) {
    if (std::fabs(t1.data()[i] - t2.data()[i]) > tol)
      return false;
  }
  return true;
}
