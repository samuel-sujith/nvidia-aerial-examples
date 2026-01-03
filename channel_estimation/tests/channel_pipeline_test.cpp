/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../channel_est_pipeline.hpp"

using namespace framework::examples;
using namespace framework::pipeline;

class ChannelEstimationPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        
        // Create basic pipeline spec
        spec_.pipeline_id = "test_channel_est_pipeline";
        spec_.batch_size = 1;
        spec_.num_slots = 1;
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    PipelineSpec spec_;
};

TEST_F(ChannelEstimationPipelineTest, Construction) {
    ChannelEstPipelineConfig config;
    config.num_tx_antennas = 4;
    config.num_rx_antennas = 4;
    config.num_subcarriers = 1024;
    config.pilot_spacing = 4;
    config.memory_pool_size = 10 * 1024 * 1024;

    ChannelEstimationPipeline pipeline("test_pipeline", config);
    EXPECT_EQ(pipeline.get_pipeline_id(), "test_pipeline");
}

TEST_F(ChannelEstimationPipelineTest, SetupAndTeardown) {
    ChannelEstPipelineConfig config;
    config.num_tx_antennas = 2;
    config.num_rx_antennas = 4;
    config.num_subcarriers = 512;
    config.pilot_spacing = 4;
    config.memory_pool_size = 5 * 1024 * 1024;

    ChannelEstimationPipeline pipeline("setup_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    EXPECT_TRUE(pipeline.is_ready());
    
    pipeline.teardown();
}

TEST_F(ChannelEstimationPipelineTest, InvalidConfiguration) {
    ChannelEstPipelineConfig config;
    config.num_tx_antennas = 0; // Invalid
    config.num_rx_antennas = 4;
    config.num_subcarriers = 1024;
    config.pilot_spacing = 4;
    config.memory_pool_size = 1024 * 1024;

    ChannelEstimationPipeline pipeline("invalid_test", config);
    EXPECT_FALSE(pipeline.setup(spec_));
}

TEST_F(ChannelEstimationPipelineTest, LargeConfiguration) {
    ChannelEstPipelineConfig config;
    config.num_tx_antennas = 8;
    config.num_rx_antennas = 8;
    config.num_subcarriers = 4096;
    config.pilot_spacing = 8;
    config.memory_pool_size = 50 * 1024 * 1024; // Large memory pool

    ChannelEstimationPipeline pipeline("large_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    pipeline.teardown();
}

TEST_F(ChannelEstimationPipelineTest, MemoryPoolSizing) {
    ChannelEstPipelineConfig config;
    config.num_tx_antennas = 4;
    config.num_rx_antennas = 4;
    config.num_subcarriers = 1024;
    config.pilot_spacing = 4;
    
    // Test insufficient memory
    config.memory_pool_size = 1024; // Very small
    ChannelEstimationPipeline small_pipeline("small_mem_test", config);
    EXPECT_FALSE(small_pipeline.setup(spec_));
    
    // Test adequate memory
    config.memory_pool_size = 10 * 1024 * 1024; // Adequate
    ChannelEstimationPipeline adequate_pipeline("adequate_mem_test", config);
    EXPECT_TRUE(adequate_pipeline.setup(spec_));
    adequate_pipeline.teardown();
}

TEST_F(ChannelEstimationPipelineTest, MultipleSetupTeardownCycles) {
    ChannelEstPipelineConfig config;
    config.num_tx_antennas = 2;
    config.num_rx_antennas = 2;
    config.num_subcarriers = 256;
    config.pilot_spacing = 4;
    config.memory_pool_size = 2 * 1024 * 1024;

    ChannelEstimationPipeline pipeline("cycle_test", config);
    
    // Test multiple setup/teardown cycles
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(pipeline.setup(spec_));
        EXPECT_TRUE(pipeline.is_ready());
        pipeline.teardown();
        EXPECT_FALSE(pipeline.is_ready());
    }
}

TEST_F(ChannelEstimationPipelineTest, StatsTracking) {
    ChannelEstPipelineConfig config;
    config.num_tx_antennas = 4;
    config.num_rx_antennas = 4;
    config.num_subcarriers = 1024;
    config.pilot_spacing = 4;
    config.memory_pool_size = 5 * 1024 * 1024;

    ChannelEstimationPipeline pipeline("stats_test", config);
    pipeline.setup(spec_);
    
    // Test stats functionality
    pipeline.print_stats(); // Should not crash
    pipeline.teardown();
}

TEST_F(ChannelEstimationPipelineTest, DifferentPilotSpacings) {
    std::vector<int> spacings = {2, 4, 8, 12};
    
    for (auto spacing : spacings) {
        ChannelEstPipelineConfig config;
        config.num_tx_antennas = 2;
        config.num_rx_antennas = 2;
        config.num_subcarriers = 1024;
        config.pilot_spacing = spacing;
        config.memory_pool_size = 5 * 1024 * 1024;

        ChannelEstimationPipeline pipeline("pilot_" + std::to_string(spacing), config);
        
        EXPECT_TRUE(pipeline.setup(spec_));
        pipeline.teardown();
    }
}