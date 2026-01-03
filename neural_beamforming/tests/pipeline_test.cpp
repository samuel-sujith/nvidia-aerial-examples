/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../beamforming_pipeline.hpp"

using namespace aerial::examples;
using namespace framework::pipeline;

class BeamformingPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        
        spec_.pipeline_id = "test_beamforming_pipeline";
        spec_.batch_size = 1;
        spec_.num_slots = 1;
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    PipelineSpec spec_;
};

TEST_F(BeamformingPipelineTest, Construction) {
    BeamformingPipelineConfig config;
    config.num_antennas = 64;
    config.num_beams = 16;
    config.algorithm = BeamformingAlgorithm::NEURAL;
    config.memory_pool_size = 10 * 1024 * 1024;

    BeamformingPipeline pipeline("test_pipeline", config);
    EXPECT_EQ(pipeline.get_pipeline_id(), "test_pipeline");
}

TEST_F(BeamformingPipelineTest, SetupAndTeardown) {
    BeamformingPipelineConfig config;
    config.num_antennas = 32;
    config.num_beams = 8;
    config.algorithm = BeamformingAlgorithm::CONVENTIONAL;
    config.memory_pool_size = 5 * 1024 * 1024;

    BeamformingPipeline pipeline("setup_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    EXPECT_TRUE(pipeline.is_ready());
    
    pipeline.teardown();
}

TEST_F(BeamformingPipelineTest, LargeAntennaArray) {
    BeamformingPipelineConfig config;
    config.num_antennas = 256;
    config.num_beams = 64;
    config.algorithm = BeamformingAlgorithm::ADAPTIVE;
    config.memory_pool_size = 50 * 1024 * 1024; // Large memory for 256 antennas

    BeamformingPipeline pipeline("large_array_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    pipeline.teardown();
}

TEST_F(BeamformingPipelineTest, NeuralNetworkPipeline) {
    BeamformingPipelineConfig config;
    config.num_antennas = 64;
    config.num_beams = 16;
    config.algorithm = BeamformingAlgorithm::NEURAL;
    config.model_path = "/tmp/neural_beamforming_model.onnx";
    config.memory_pool_size = 20 * 1024 * 1024;

    BeamformingPipeline pipeline("neural_test", config);
    
    // Neural network models might not be available in test environment
    // So we just test construction and basic setup
    EXPECT_NO_THROW(pipeline.setup(spec_));
    pipeline.teardown();
}

TEST_F(BeamformingPipelineTest, MultipleBeamConfigurations) {
    std::vector<std::pair<int, int>> configs = {
        {16, 4}, {32, 8}, {64, 16}, {128, 32}
    };
    
    for (auto [antennas, beams] : configs) {
        BeamformingPipelineConfig config;
        config.num_antennas = antennas;
        config.num_beams = beams;
        config.algorithm = BeamformingAlgorithm::CONVENTIONAL;
        config.memory_pool_size = antennas * 1024 * 100; // Scale memory with antennas

        BeamformingPipeline pipeline("multi_" + std::to_string(antennas) + "x" + std::to_string(beams), config);
        
        EXPECT_TRUE(pipeline.setup(spec_));
        pipeline.teardown();
    }
}

TEST_F(BeamformingPipelineTest, StatsTracking) {
    BeamformingPipelineConfig config;
    config.num_antennas = 32;
    config.num_beams = 8;
    config.algorithm = BeamformingAlgorithm::NEURAL;
    config.memory_pool_size = 5 * 1024 * 1024;

    BeamformingPipeline pipeline("stats_test", config);
    pipeline.setup(spec_);
    
    pipeline.print_stats(); // Should not crash
}