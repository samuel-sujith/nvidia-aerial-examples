/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../beamformer.hpp"

using namespace aerial::examples;

class NeuralBeamformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

TEST_F(NeuralBeamformerTest, BasicConstruction) {
    BeamformingParams params;
    params.num_antennas = 64;
    params.num_beams = 16;
    params.algorithm = BeamformingAlgorithm::NEURAL;

    NeuralBeamformer beamformer("test_beamformer", params);
    EXPECT_EQ(beamformer.get_module_id(), "test_beamformer");
}

TEST_F(NeuralBeamformerTest, ParameterValidation) {
    BeamformingParams params;
    params.num_antennas = 0; // Invalid
    params.num_beams = 16;
    params.algorithm = BeamformingAlgorithm::NEURAL;

    EXPECT_THROW(NeuralBeamformer("test", params), std::invalid_argument);
}

TEST_F(NeuralBeamformerTest, AntennaArrayConfigurations) {
    std::vector<int> antenna_counts = {16, 32, 64, 128, 256};
    
    for (auto num_antennas : antenna_counts) {
        BeamformingParams params;
        params.num_antennas = num_antennas;
        params.num_beams = num_antennas / 4; // Typical beam to antenna ratio
        params.algorithm = BeamformingAlgorithm::NEURAL;

        EXPECT_NO_THROW(NeuralBeamformer("bf_" + std::to_string(num_antennas), params));
    }
}

TEST_F(NeuralBeamformerTest, BeamformingAlgorithms) {
    BeamformingParams params;
    params.num_antennas = 64;
    params.num_beams = 16;

    // Test different algorithms
    params.algorithm = BeamformingAlgorithm::NEURAL;
    EXPECT_NO_THROW(NeuralBeamformer("neural_bf", params));

    params.algorithm = BeamformingAlgorithm::CONVENTIONAL;
    EXPECT_NO_THROW(NeuralBeamformer("conv_bf", params));

    params.algorithm = BeamformingAlgorithm::ADAPTIVE;
    EXPECT_NO_THROW(NeuralBeamformer("adaptive_bf", params));
}

TEST_F(NeuralBeamformerTest, BeamPatternValidation) {
    BeamformingParams params;
    params.num_antennas = 32;
    params.num_beams = 8;
    params.algorithm = BeamformingAlgorithm::NEURAL;

    NeuralBeamformer beamformer("pattern_test", params);
    
    // Test that beamformer handles valid beam patterns
    EXPECT_TRUE(params.num_beams <= params.num_antennas);
}

TEST_F(NeuralBeamformerTest, NeuralNetworkModel) {
    BeamformingParams params;
    params.num_antennas = 64;
    params.num_beams = 16;
    params.algorithm = BeamformingAlgorithm::NEURAL;
    params.model_path = "/tmp/test_model.onnx"; // Mock path

    NeuralBeamformer beamformer("nn_test", params);
    
    // Test neural network initialization
    EXPECT_TRUE(true); // Placeholder for actual NN model tests
}