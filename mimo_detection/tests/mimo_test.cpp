/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../mimo_detector.hpp"

using namespace aerial::examples;

class MIMODetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

TEST_F(MIMODetectorTest, BasicConstruction) {
    MIMOParams params;
    params.num_tx_antennas = 4;
    params.num_rx_antennas = 4;
    params.detection_algorithm = MIMOAlgorithm::ZF;

    MIMODetector detector("test_mimo", params);
    EXPECT_EQ(detector.get_module_id(), "test_mimo");
}

TEST_F(MIMODetectorTest, ParameterValidation) {
    MIMOParams params;
    params.num_tx_antennas = 0; // Invalid
    params.num_rx_antennas = 4;
    params.detection_algorithm = MIMOAlgorithm::ZF;

    EXPECT_THROW(MIMODetector("test", params), std::invalid_argument);
}

TEST_F(MIMODetectorTest, AntennaConfigurations) {
    std::vector<std::pair<int, int>> configs = {
        {2, 2}, {2, 4}, {4, 4}, {4, 8}, {8, 8}
    };
    
    for (auto [tx, rx] : configs) {
        MIMOParams params;
        params.num_tx_antennas = tx;
        params.num_rx_antennas = rx;
        params.detection_algorithm = MIMOAlgorithm::MMSE;

        EXPECT_NO_THROW(MIMODetector("mimo_" + std::to_string(tx) + "x" + std::to_string(rx), params));
    }
}

TEST_F(MIMODetectorTest, DetectionAlgorithms) {
    MIMOParams params;
    params.num_tx_antennas = 4;
    params.num_rx_antennas = 4;

    // Test different algorithms
    params.detection_algorithm = MIMOAlgorithm::ZF;
    EXPECT_NO_THROW(MIMODetector("zf_detector", params));

    params.detection_algorithm = MIMOAlgorithm::MMSE;
    EXPECT_NO_THROW(MIMODetector("mmse_detector", params));

    params.detection_algorithm = MIMOAlgorithm::ML;
    EXPECT_NO_THROW(MIMODetector("ml_detector", params));
}

TEST_F(MIMODetectorTest, ChannelMatrixValidation) {
    MIMOParams params;
    params.num_tx_antennas = 2;
    params.num_rx_antennas = 2;
    params.detection_algorithm = MIMOAlgorithm::ZF;

    MIMODetector detector("channel_test", params);
    
    // Test that detector can handle basic 2x2 MIMO
    EXPECT_TRUE(true); // Placeholder for actual channel matrix tests
}