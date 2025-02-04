#include "func.h"
#include <gtest/gtest.h>

// Test fixture for the Calculator class
class FunctionalTests : public ::testing::Test {
protected:
};

TEST_F(FunctionalTests, AddNumbers) { EXPECT_EQ(synapse::add(2, 3), 5); }

TEST_F(FunctionalTests, MulNumbers) { EXPECT_EQ(synapse::mul(-2, -3), 6); }
