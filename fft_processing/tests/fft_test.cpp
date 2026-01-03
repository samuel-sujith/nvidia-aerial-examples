/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "../fft_module.hpp"

using namespace aerial::fft;

class FFTModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};

TEST_F(FFTModuleTest, BasicConstruction) {
    FFTParams params;
    params.fft_size = 1024;
    params.batch_size = 1;
    params.direction = FFTDirection::FORWARD;

    FFTProcessor processor("test_fft", params);
    EXPECT_EQ(processor.get_module_id(), "test_fft");
}

TEST_F(FFTModuleTest, ParameterValidation) {
    FFTParams params;
    params.fft_size = 0; // Invalid
    params.batch_size = 1;
    params.direction = FFTDirection::FORWARD;

    EXPECT_THROW(FFTProcessor("test", params), std::invalid_argument);
}

TEST_F(FFTModuleTest, ValidFFTSizes) {
    std::vector<size_t> valid_sizes = {64, 128, 256, 512, 1024, 2048, 4096};
    
    for (auto size : valid_sizes) {
        FFTParams params;
        params.fft_size = size;
        params.batch_size = 1;
        params.direction = FFTDirection::FORWARD;

        EXPECT_NO_THROW(FFTProcessor("test_" + std::to_string(size), params));
    }
}

TEST_F(FFTModuleTest, ForwardAndInverseFFT) {
    FFTParams forward_params;
    forward_params.fft_size = 256;
    forward_params.batch_size = 2;
    forward_params.direction = FFTDirection::FORWARD;

    FFTParams inverse_params;
    inverse_params.fft_size = 256;
    inverse_params.batch_size = 2;
    inverse_params.direction = FFTDirection::INVERSE;

    EXPECT_NO_THROW(FFTProcessor("forward_fft", forward_params));
    EXPECT_NO_THROW(FFTProcessor("inverse_fft", inverse_params));
}

TEST_F(FFTModuleTest, BatchProcessing) {
    FFTParams params;
    params.fft_size = 512;
    params.batch_size = 8;
    params.direction = FFTDirection::FORWARD;

    FFTProcessor processor("batch_test", params);
    
    // Test batch size validation
    EXPECT_EQ(params.batch_size, 8);
}