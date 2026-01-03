/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../modulation_pipeline.hpp"

using namespace modulation;
using namespace framework::pipeline;

class ModulationPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        
        // Create basic pipeline spec
        spec_.pipeline_id = "test_modulation_pipeline";
        spec_.batch_size = 1;
        spec_.num_slots = 1;
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    PipelineSpec spec_;
};

TEST_F(ModulationPipelineTest, Construction) {
    ModulationPipelineConfig config;
    config.modulation_scheme = aerial::examples::ModulationScheme::QPSK;
    config.memory_pool_size = 1024 * 1024;

    ModulationPipeline pipeline("test_pipeline", config);
    EXPECT_EQ(pipeline.get_pipeline_id(), "test_pipeline");
}

TEST_F(ModulationPipelineTest, SetupAndTeardown) {
    ModulationPipelineConfig config;
    config.modulation_scheme = aerial::examples::ModulationScheme::QAM16;
    config.memory_pool_size = 2 * 1024 * 1024;

    ModulationPipeline pipeline("setup_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    EXPECT_TRUE(pipeline.is_ready());
    
    pipeline.teardown();
}

TEST_F(ModulationPipelineTest, InvalidConfig) {
    ModulationPipelineConfig config;
    config.modulation_scheme = aerial::examples::ModulationScheme::QAM16;
    config.memory_pool_size = 0; // Invalid size

    ModulationPipeline pipeline("invalid_test", config);
    EXPECT_FALSE(pipeline.setup(spec_));
}

TEST_F(ModulationPipelineTest, StatsTracking) {
    ModulationPipelineConfig config;
    config.modulation_scheme = aerial::examples::ModulationScheme::QPSK;
    config.memory_pool_size = 1024 * 1024;

    ModulationPipeline pipeline("stats_test", config);
    pipeline.setup(spec_);
    
    // Test that stats tracking works
    pipeline.print_stats(); // Should not crash
}