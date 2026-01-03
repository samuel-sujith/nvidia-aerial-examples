/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../mimo_pipeline.hpp"

using namespace aerial::examples;
using namespace framework::pipeline;

class MIMOPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        
        spec_.pipeline_id = "test_mimo_pipeline";
        spec_.batch_size = 1;
        spec_.num_slots = 1;
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    PipelineSpec spec_;
};

TEST_F(MIMOPipelineTest, Construction) {
    MIMOPipelineConfig config;
    config.num_tx_antennas = 4;
    config.num_rx_antennas = 4;
    config.detection_algorithm = MIMOAlgorithm::MMSE;
    config.memory_pool_size = 2 * 1024 * 1024;

    MIMOPipeline pipeline("test_pipeline", config);
    EXPECT_EQ(pipeline.get_pipeline_id(), "test_pipeline");
}

TEST_F(MIMOPipelineTest, SetupAndTeardown) {
    MIMOPipelineConfig config;
    config.num_tx_antennas = 2;
    config.num_rx_antennas = 4;
    config.detection_algorithm = MIMOAlgorithm::ZF;
    config.memory_pool_size = 4 * 1024 * 1024;

    MIMOPipeline pipeline("setup_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    EXPECT_TRUE(pipeline.is_ready());
    
    pipeline.teardown();
}

TEST_F(MIMOPipelineTest, InvalidAntennaConfig) {
    MIMOPipelineConfig config;
    config.num_tx_antennas = 8;
    config.num_rx_antennas = 2; // Rx < Tx not typical
    config.detection_algorithm = MIMOAlgorithm::ZF;
    config.memory_pool_size = 1024 * 1024;

    MIMOPipeline pipeline("invalid_test", config);
    // Some configurations might be valid depending on implementation
    // This test just ensures no crash occurs
    EXPECT_NO_THROW(pipeline.setup(spec_));
}

TEST_F(MIMOPipelineTest, LargeMIMOConfiguration) {
    MIMOPipelineConfig config;
    config.num_tx_antennas = 8;
    config.num_rx_antennas = 8;
    config.detection_algorithm = MIMOAlgorithm::MMSE;
    config.memory_pool_size = 16 * 1024 * 1024; // Larger memory for 8x8

    MIMOPipeline pipeline("large_mimo_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    pipeline.teardown();
}

TEST_F(MIMOPipelineTest, StatsTracking) {
    MIMOPipelineConfig config;
    config.num_tx_antennas = 4;
    config.num_rx_antennas = 4;
    config.detection_algorithm = MIMOAlgorithm::ZF;
    config.memory_pool_size = 2 * 1024 * 1024;

    MIMOPipeline pipeline("stats_test", config);
    pipeline.setup(spec_);
    
    pipeline.print_stats(); // Should not crash
}