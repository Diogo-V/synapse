cmake_minimum_required(VERSION 3.15...3.27)
project(synapse)

# Helps clangd find the headers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sets the c++ required version
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(FLAGS 
  -Wall -Weffc++ -Wextra -Wundef -Wshadow -Wcast-align -Wpointer-arith 
  -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual 
  -Wswitch-default -Wswitch-enum -Wunreachable-code -Wconversion
  -Werror -pedantic-errors
)

# Add the required header files to the compiler
include_directories(${CMAKE_SOURCE_DIR}/synapse/include)

# Add the source code and the test code
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/synapse/*.cpp)
file(GLOB_RECURSE TEST_FILES ${CMAKE_SOURCE_DIR}/tests/*.cpp)

# Create a static library for synapse
add_library(synapse STATIC ${SRC_FILES})
target_compile_options(synapse PRIVATE ${FLAGS})

# Include Google Test for testing
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}/include)

# Create a test executable
add_executable(synapse_tests ${TEST_FILES})
target_compile_options(synapse_tests PRIVATE ${FLAGS})

# Link the library with the test code
target_link_libraries(synapse_tests synapse ${GTEST_MAIN_LIBRARIES} Threads::Threads)

# Add the test to CTest
enable_testing()
add_test(NAME synapse_tests COMMAND synapse_tests)
