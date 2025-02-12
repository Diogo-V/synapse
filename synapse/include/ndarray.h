#ifndef NDARRAY_H
#define NDARRAY_H

#include <cstddef>
#include <string>
#include <vector>

namespace synapse {

// Auxiliary types
using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

// Converts an N dimensional index into a position in the vector of data
size_t nd_index_to_pos(Shape indices, Strides strides);

// Converts a position in a vector of data into an N dimensional index
Shape pos_to_nd_index(size_t pos, Shape shape);

class NDArray {
public:
  NDArray(std::vector<float> data, Shape shape);
  ~NDArray();

  // Accessors
  const Shape &shape() const;
  const Strides &strides() const;
  std::vector<float> &data();
  const std::vector<float> &data() const;
  size_t ndim() const;
  size_t size() const;

  // Methods
  bool is_contigous();
  const std::string to_string() const;

  // Allows accessing elements of the ndarray directly
  template <typename... Indices>
  const float &operator()(Indices... indices) const {
    return this->data()[_operator_parenthesis(indices...)];
  }
  template <typename... Indices> float &operator()(Indices... indices) {
    return this->data()[_operator_parenthesis(indices...)];
  }

private:
  std::vector<float> _data;
  Shape _shape;
  Strides _strides;
  size_t _ndim;
  size_t _size;

  template <typename... Indices>
  size_t _operator_parenthesis(Indices... indices) const {
    static_assert(sizeof...(indices) > 0, "At least one index is required.");
    Shape idx_vec{static_cast<size_t>(indices)...};

    // Bounds checking
    if (idx_vec.size() != this->ndim()) {
      throw std::out_of_range(
          "Number of indices does not match the number of dimensions.");
    }
    for (size_t i = 0; i < idx_vec.size(); i++) {
      if (idx_vec[i] >= this->shape()[i]) {
        throw std::out_of_range("Index out of bounds for dimension " +
                                std::to_string(i));
      }
    }

    return nd_index_to_pos(idx_vec, this->strides());
  }
};
} // namespace synapse

#endif // !NDARRAY_H
