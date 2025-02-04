#ifndef NDARRAY_H
#define NDARRAY_H

#include <cstddef>
#include <vector>

namespace synapse {
class NDArray {
public:
  std::vector<float> data;
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  size_t ndim;
  size_t size;

  NDArray(std::vector<float> data, std::vector<size_t> shape);
  ~NDArray();
};
} // namespace synapse

#endif // !NDARRAY_H
