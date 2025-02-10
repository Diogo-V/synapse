#ifndef NDARRAY_H
#define NDARRAY_H

#include <cstddef>
#include <string>
#include <vector>

namespace synapse {
class NDArray {
public:
  NDArray(std::vector<float> data, std::vector<size_t> shape);
  ~NDArray();

  const std::vector<size_t> &shape() const;
  std::vector<float> &data();
  const std::vector<float> &data() const;
  size_t ndim() const;
  size_t size() const;

  bool is_contigous();
  const std::string to_string() const;

  // Allows accessing elements of the ndarray directly
  template <typename... Indices>
  const float &operator()(Indices... indices) const;
  template <typename... Indices> float &operator()(Indices... indices);

private:
  std::vector<float> _data;
  std::vector<size_t> _shape;
  std::vector<size_t> _strides;
  size_t _ndim;
  size_t _size;
};

// Converts an N dimensional index into a position in the vector of data
size_t nd_index_to_pos(std::vector<size_t> indices,
                       std::vector<size_t> strides);

// Converts a position in a vector of data into an N dimensional index
std::vector<size_t> pos_to_nd_index(size_t pos, std::vector<size_t> shape);

} // namespace synapse

#endif // !NDARRAY_H
