cmake_minimum_required(VERSION 3.15...3.27)
project(synapse)

# This file is meant to be ran by scikit-build-core
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

# Helps clangd find the headers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sets the c++ required version
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Finds required python version and required CMake version
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# This lines force nanobind to always build an optimized 'release' executable. Without
# them, the binding code may run slowly and increase ovcerhead
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Add the required header files to the compiler
include_directories(${CMAKE_SOURCE_DIR}/synapse/csrc/include)
include_directories(${nanobind_SOURCE_DIR}/include)
include_directories(${Python_INCLUDE_DIRS})

# Finds the C++ sources
file(GLOB SRC_FILES synapse/csrc/*.cpp)

# Add the bindings
nanobind_add_module(_synapse ${SRC_FILES})

# Set compiler flags
set(FLAGS 
  -Wall -Weffc++ -Wextra -Wundef -Wshadow -Wcast-align -Wpointer-arith 
  -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual 
  -Wswitch-default -Wswitch-enum -Wunreachable-code -Wconversion
  # -Werror -pedantic-errors
)
target_compile_options(_synapse PRIVATE ${FLAGS})

# Install the Python module
install(TARGETS _synapse LIBRARY DESTINATION .)

