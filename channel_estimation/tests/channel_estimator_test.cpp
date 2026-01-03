/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../channel_estimator.hpp"

using namespace framework::examples;

class ChannelEstimatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context
        cudaSetDevice(0);
    }

    void TearDown() override {
        // Clean up CUDA context
        cudaDeviceReset();
    }
};

TEST_F(ChannelEstimatorTest, BasicConstruction) {
    ChannelEstParams params;
    params.num_tx_antennas = 4;
    params.num_rx_antennas = 4;
    params.num_subcarriers = 1024;
    params.pilot_spacing = 4;

    ChannelEstimator estimator("test_estimator", params);
    EXPECT_EQ(estimator.get_module_id(), "test_estimator");
}

TEST_F(ChannelEstimatorTest, ParameterValidation) {
    ChannelEstParams params;
    params.num_tx_antennas = 0; // Invalid
    params.num_rx_antennas = 4;
    params.num_subcarriers = 1024;
    params.pilot_spacing = 4;

    EXPECT_THROW(ChannelEstimator("invalid_test", params), std::invalid_argument);
}

TEST_F(ChannelEstimatorTest, AntennaConfigurations) {
    std::vector<std::pair<int, int>> antenna_configs = {
        {1, 1}, {2, 2}, {2, 4}, {4, 4}, {4, 8}, {8, 8}
    };
    
    for (auto [tx, rx] : antenna_configs) {
        ChannelEstParams params;
        params.num_tx_antennas = tx;
        params.num_rx_antennas = rx;
        params.num_subcarriers = 512;
        params.pilot_spacing = 4;

        EXPECT_NO_THROW(ChannelEstimator("est_" + std::to_string(tx) + "x" + std::to_string(rx), params));
    }
}

TEST_F(ChannelEstimatorTest, SubcarrierValidation) {
    std::vector<int> subcarrier_counts = {64, 128, 256, 512, 1024, 2048};
    
    for (auto num_sc : subcarrier_counts) {
        ChannelEstParams params;
        params.num_tx_antennas = 2;
        params.num_rx_antennas = 2;
        params.num_subcarriers = num_sc;
        params.pilot_spacing = 4;

        EXPECT_NO_THROW(ChannelEstimator("sc_" + std::to_string(num_sc), params));
    }
}

TEST_F(ChannelEstimatorTest, PilotSpacingValidation) {
    std::vector<int> pilot_spacings = {2, 4, 8, 12, 16};
    
    for (auto spacing : pilot_spacings) {
        ChannelEstParams params;
        params.num_tx_antennas = 2;
        params.num_rx_antennas = 2;
        params.num_subcarriers = 1024;
        params.pilot_spacing = spacing;

        EXPECT_NO_THROW(ChannelEstimator("pilot_" + std::to_string(spacing), params));
    }
}

TEST_F(ChannelEstimatorTest, EstimationAlgorithms) {
    ChannelEstParams params;
    params.num_tx_antennas = 4;
    params.num_rx_antennas = 4;
    params.num_subcarriers = 1024;
    params.pilot_spacing = 4;

    // Test different estimation algorithms if available
    params.estimation_method = ChannelEstimationMethod::LS; // Least Squares
    EXPECT_NO_THROW(ChannelEstimator("ls_estimator", params));

    params.estimation_method = ChannelEstimationMethod::MMSE; // Minimum Mean Square Error
    EXPECT_NO_THROW(ChannelEstimator("mmse_estimator", params));
}

TEST_F(ChannelEstimatorTest, MemoryRequirements) {
    ChannelEstParams params;
    params.num_tx_antennas = 8;
    params.num_rx_antennas = 8;
    params.num_subcarriers = 2048;
    params.pilot_spacing = 4;

    ChannelEstimator estimator("memory_test", params);
    
    // Test that large configurations can be constructed
    // Memory allocation testing would be done in integration tests
    EXPECT_TRUE(true);
}