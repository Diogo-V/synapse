#include "func.h"
#include <nanobind/nanobind.h>

NB_MODULE(_synapse, m) { m.def("add", &synapse::add); }
