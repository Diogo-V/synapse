#include "func.h"
#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

// Test fixture for the Calculator class
class FunctionalTests : public ::testing::Test {
protected:
  synapse::Tensor t1{std::vector<float>{0.2f, 0.5f, 46.0f, -5.1f},
                     synapse::Shape{4}};
  synapse::Tensor t2{std::vector<float>{0.2f, 0.5f, 46.0f, -5.1f},
                     synapse::Shape{4}};
};

TEST_F(FunctionalTests, AddTensors) {
  synapse::Tensor result = synapse::add(t1, t2);
  synapse::Tensor expected_tensor{std::vector<float>{0.4f, 1.0f, 92.0f, -10.2f},
                                  synapse::Shape{4}};
  EXPECT_TRUE(synapse::is_close(result, expected_tensor));
}

TEST_F(FunctionalTests, MulTensors) {
  synapse::Tensor result = synapse::mul(t1, t2);
  synapse::Tensor expected_tensor{
      std::vector<float>{0.04f, 0.25f, 2116.0f, 26.01f}, synapse::Shape{4}};
  EXPECT_TRUE(synapse::is_close(result, expected_tensor));
}

TEST_F(FunctionalTests, AddTensorsMismatchShape) {
  synapse::Tensor t3{std::vector<float>{1.0f, 2.0f}, synapse::Shape{2}};
  EXPECT_THROW(synapse::add(t1, t3), std::invalid_argument);
}

TEST_F(FunctionalTests, MulTensorsMismatchShape) {
  synapse::Tensor t3{std::vector<float>{1.0f, 2.0f}, synapse::Shape{2}};
  EXPECT_THROW(synapse::mul(t1, t3), std::invalid_argument);
}
