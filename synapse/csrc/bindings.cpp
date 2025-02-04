#include "func.h"
#include "ndarray.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_synapse, m) {
  // Main structure definition
  nb::class_<synapse::NDArray>(m, "NDArray")
      .def(nb::init<std::vector<float>, std::vector<size_t>>())
      .def_ro("ndim", &synapse::NDArray::ndim)
      .def_ro("size", &synapse::NDArray::size)
      .def_ro("data", &synapse::NDArray::data)
      .def_ro("shape", &synapse::NDArray::shape)
      .def_ro("strides", &synapse::NDArray::strides);

  // Ops definition
  m.def("add", &synapse::add, "Adds two numbers together");
  m.def("mul", &synapse::mul, "Multiplies two numbers together");
}
