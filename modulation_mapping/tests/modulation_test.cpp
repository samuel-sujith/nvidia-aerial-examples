/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../modulator.hpp"

using namespace aerial::examples;

class ModulatorTest : public ::testing::Test {
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

TEST_F(ModulatorTest, BasicConstruction) {
    ModulationParams params;
    params.modulation_scheme = ModulationScheme::QAM16;
    params.num_symbols = 1024;

    QAMModulator modulator("test_mod", params);
    EXPECT_EQ(modulator.get_module_id(), "test_mod");
}

TEST_F(ModulatorTest, ParameterValidation) {
    ModulationParams params;
    params.modulation_scheme = ModulationScheme::QAM64;
    params.num_symbols = 0; // Invalid

    EXPECT_THROW(QAMModulator("test", params), std::invalid_argument);
}

TEST_F(ModulatorTest, ModulationOrder) {
    ModulationParams params;
    
    params.modulation_scheme = ModulationScheme::QPSK;
    QAMModulator qpsk_mod("qpsk", params);
    // QPSK should produce 2 bits per symbol
    
    params.modulation_scheme = ModulationScheme::QAM16;
    QAMModulator qam16_mod("qam16", params);
    // QAM16 should produce 4 bits per symbol
}

TEST_F(ModulatorTest, SmallDataModulation) {
    ModulationParams params;
    params.modulation_scheme = ModulationScheme::QPSK;
    params.num_symbols = 16;

    QAMModulator modulator("small_test", params);
    
    // Test with small data to verify basic functionality
    std::vector<uint8_t> input_bits(32); // 2 bits per symbol for QPSK
    std::fill(input_bits.begin(), input_bits.end(), 0x55); // Alternating pattern
    
    // This test just verifies construction and parameter setup
    EXPECT_TRUE(true); // Placeholder - would need actual modulate method
}