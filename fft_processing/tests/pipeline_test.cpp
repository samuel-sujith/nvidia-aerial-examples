/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../fft_pipeline.hpp"

using namespace aerial::fft;
using namespace framework::pipeline;

class FFTPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        
        spec_.pipeline_id = "test_fft_pipeline";
        spec_.batch_size = 1;
        spec_.num_slots = 1;
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    PipelineSpec spec_;
};

TEST_F(FFTPipelineTest, Construction) {
    FFTPipelineConfig config;
    config.fft_size = 1024;
    config.batch_size = 1;
    config.direction = FFTDirection::FORWARD;
    config.memory_pool_size = 1024 * 1024;

    FFTPipeline pipeline("test_pipeline", config);
    EXPECT_EQ(pipeline.get_pipeline_id(), "test_pipeline");
}

TEST_F(FFTPipelineTest, SetupAndTeardown) {
    FFTPipelineConfig config;
    config.fft_size = 512;
    config.batch_size = 2;
    config.direction = FFTDirection::FORWARD;
    config.memory_pool_size = 2 * 1024 * 1024;

    FFTPipeline pipeline("setup_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    EXPECT_TRUE(pipeline.is_ready());
    
    pipeline.teardown();
}

TEST_F(FFTPipelineTest, InvalidFFTSize) {
    FFTPipelineConfig config;
    config.fft_size = 0; // Invalid
    config.batch_size = 1;
    config.direction = FFTDirection::FORWARD;
    config.memory_pool_size = 1024 * 1024;

    FFTPipeline pipeline("invalid_test", config);
    EXPECT_FALSE(pipeline.setup(spec_));
}

TEST_F(FFTPipelineTest, MemoryManagement) {
    FFTPipelineConfig config;
    config.fft_size = 2048;
    config.batch_size = 4;
    config.direction = FFTDirection::INVERSE;
    config.memory_pool_size = 10 * 1024 * 1024; // Large pool

    FFTPipeline pipeline("memory_test", config);
    
    EXPECT_TRUE(pipeline.setup(spec_));
    
    // Test multiple setup/teardown cycles
    pipeline.teardown();
    EXPECT_TRUE(pipeline.setup(spec_));
    pipeline.teardown();
}

TEST_F(FFTPipelineTest, StatsTracking) {
    FFTPipelineConfig config;
    config.fft_size = 256;
    config.batch_size = 1;
    config.direction = FFTDirection::FORWARD;
    config.memory_pool_size = 1024 * 1024;

    FFTPipeline pipeline("stats_test", config);
    pipeline.setup(spec_);
    
    pipeline.print_stats(); // Should not crash
}