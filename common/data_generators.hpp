#pragma once

#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace data_generators {

/// Complex number type alias
using ComplexFloat = std::complex<float>;

/// Random number generator with fixed seed for reproducibility
class RandomGenerator {
private:
    mutable std::mt19937 gen_;
    mutable std::uniform_real_distribution<float> uniform_dist_;
    mutable std::normal_distribution<float> normal_dist_;
    
public:
    explicit RandomGenerator(uint32_t seed = 42) 
        : gen_(seed), uniform_dist_(0.0f, 1.0f), normal_dist_(0.0f, 1.0f) {}
    
    /// Generate uniform random float in [0, 1)
    float uniform() const {
        return uniform_dist_(gen_);
    }
    
    /// Generate uniform random float in [min, max)
    float uniform(float min_val, float max_val) const {
        return min_val + (max_val - min_val) * uniform();
    }
    
    /// Generate normal random float with mean 0 and std 1
    float normal() const {
        return normal_dist_(gen_);
    }
    
    /// Generate normal random float with specified mean and standard deviation
    float normal(float mean, float std_dev) const {
        return mean + std_dev * normal();
    }
    
    /// Generate complex number with normal distribution
    ComplexFloat complex_normal(float mean = 0.0f, float std_dev = 1.0f) const {
        return ComplexFloat(normal(mean, std_dev), normal(mean, std_dev));
    }
    
    /// Reset generator with new seed
    void set_seed(uint32_t seed) {
        gen_.seed(seed);
    }
};

/// Generate 5G NR-like channel estimation data
class ChannelDataGenerator {
private:
    RandomGenerator rng_;
    
public:
    explicit ChannelDataGenerator(uint32_t seed = 42) : rng_(seed) {}
    
    /// Generate OFDM pilot symbols
    std::vector<ComplexFloat> generate_pilots(size_t num_pilots, float pilot_power = 1.0f) const {
        std::vector<ComplexFloat> pilots;
        pilots.reserve(num_pilots);
        
        // Generate QPSK pilots (typical for 5G NR)
        for (size_t i = 0; i < num_pilots; ++i) {
            float real = (rng_.uniform() > 0.5f) ? pilot_power : -pilot_power;
            float imag = (rng_.uniform() > 0.5f) ? pilot_power : -pilot_power;
            pilots.emplace_back(real / std::sqrt(2.0f), imag / std::sqrt(2.0f));
        }
        
        return pilots;
    }
    
    /// Generate received pilot symbols with channel and noise
    std::vector<ComplexFloat> generate_received_pilots(
        const std::vector<ComplexFloat>& transmitted_pilots,
        const std::vector<ComplexFloat>& channel_response,
        float noise_power = 0.01f) const {
        
        assert(transmitted_pilots.size() == channel_response.size());
        
        std::vector<ComplexFloat> received;
        received.reserve(transmitted_pilots.size());
        
        float noise_std = std::sqrt(noise_power / 2.0f); // Complex noise
        
        for (size_t i = 0; i < transmitted_pilots.size(); ++i) {
            ComplexFloat noise = rng_.complex_normal(0.0f, noise_std);
            received.push_back(transmitted_pilots[i] * channel_response[i] + noise);
        }
        
        return received;
    }
    
    /// Generate realistic wireless channel response
    std::vector<ComplexFloat> generate_channel_response(
        size_t num_subcarriers,
        size_t num_paths = 6,
        float max_delay_spread = 1e-6f,
        float symbol_duration = 71.4e-6f) const {
        
        std::vector<ComplexFloat> channel(num_subcarriers, ComplexFloat(0.0f, 0.0f));
        
        // Generate multipath channel
        for (size_t path = 0; path < num_paths; ++path) {
            // Path delay (normalized to symbol duration)
            float delay = rng_.uniform() * max_delay_spread / symbol_duration;
            
            // Path gain with exponential power delay profile
            float path_power = std::exp(-3.0f * delay);
            float path_amplitude = std::sqrt(path_power);
            
            // Random phase for each path
            float phase = rng_.uniform() * 2.0f * M_PI;
            ComplexFloat path_gain(
                path_amplitude * std::cos(phase),
                path_amplitude * std::sin(phase)
            );
            
            // Apply frequency response for this path
            for (size_t k = 0; k < num_subcarriers; ++k) {
                float freq_phase = -2.0f * M_PI * delay * k / num_subcarriers;
                ComplexFloat freq_response(std::cos(freq_phase), std::sin(freq_phase));
                channel[k] += path_gain * freq_response;
            }
        }
        
        // Normalize channel power
        float total_power = 0.0f;
        for (const auto& h : channel) {
            total_power += std::norm(h);
        }
        float normalization = std::sqrt(static_cast<float>(num_subcarriers) / total_power);
        
        for (auto& h : channel) {
            h *= normalization;
        }
        
        return channel;
    }
    
    /// Generate frequency-selective fading channel
    std::vector<ComplexFloat> generate_fading_channel(
        size_t num_subcarriers,
        float doppler_freq = 100.0f,  // Hz
        float sample_rate = 30.72e6f) const {
        
        std::vector<ComplexFloat> channel;
        channel.reserve(num_subcarriers);
        
        // Jake's model for Rayleigh fading
        for (size_t i = 0; i < num_subcarriers; ++i) {
            float freq_offset = (static_cast<float>(i) / num_subcarriers - 0.5f) * sample_rate;
            
            // Doppler effect
            float doppler_shift = doppler_freq * std::cos(2.0f * M_PI * rng_.uniform());
            float effective_freq = freq_offset + doppler_shift;
            
            // Complex channel gain
            float magnitude = std::sqrt(-2.0f * std::log(rng_.uniform()));
            float phase = 2.0f * M_PI * rng_.uniform();
            
            channel.emplace_back(
                magnitude * std::cos(phase),
                magnitude * std::sin(phase)
            );
        }
        
        return channel;
    }
};

/// Generate modulation and demodulation test data
class ModulationDataGenerator {
private:
    RandomGenerator rng_;
    
public:
    explicit ModulationDataGenerator(uint32_t seed = 42) : rng_(seed) {}
    
    /// Generate random bit stream
    std::vector<uint8_t> generate_bits(size_t num_bits) const {
        std::vector<uint8_t> bits;
        bits.reserve(num_bits);
        
        for (size_t i = 0; i < num_bits; ++i) {
            bits.push_back(rng_.uniform() > 0.5f ? 1 : 0);
        }
        
        return bits;
    }
    
    /// Generate QAM constellation symbols
    std::vector<ComplexFloat> generate_qam_symbols(
        const std::vector<uint8_t>& bits,
        int modulation_order = 16) const {
        
        int bits_per_symbol = static_cast<int>(std::log2(modulation_order));
        assert(bits.size() % bits_per_symbol == 0);
        
        std::vector<ComplexFloat> symbols;
        symbols.reserve(bits.size() / bits_per_symbol);
        
        for (size_t i = 0; i < bits.size(); i += bits_per_symbol) {
            int symbol_value = 0;
            for (int j = 0; j < bits_per_symbol; ++j) {
                symbol_value |= (bits[i + j] << (bits_per_symbol - 1 - j));
            }
            
            ComplexFloat symbol = qam_map_symbol(symbol_value, modulation_order);
            symbols.push_back(symbol);
        }
        
        return symbols;
    }
    
    /// Generate noisy received symbols
    std::vector<ComplexFloat> add_awgn_noise(
        const std::vector<ComplexFloat>& symbols,
        float snr_db) const {
        
        float noise_power = std::pow(10.0f, -snr_db / 10.0f);
        float noise_std = std::sqrt(noise_power / 2.0f);
        
        std::vector<ComplexFloat> noisy_symbols;
        noisy_symbols.reserve(symbols.size());
        
        for (const auto& symbol : symbols) {
            ComplexFloat noise = rng_.complex_normal(0.0f, noise_std);
            noisy_symbols.push_back(symbol + noise);
        }
        
        return noisy_symbols;
    }
    
private:
    /// Map symbol value to QAM constellation point
    ComplexFloat qam_map_symbol(int symbol_value, int modulation_order) const {
        int sqrt_order = static_cast<int>(std::sqrt(modulation_order));
        
        int i_value = symbol_value / sqrt_order;
        int q_value = symbol_value % sqrt_order;
        
        float i_symbol = static_cast<float>(i_value) - (sqrt_order - 1) / 2.0f;
        float q_symbol = static_cast<float>(q_value) - (sqrt_order - 1) / 2.0f;
        
        // Normalize for unit average power
        float normalization = std::sqrt(2.0f * (modulation_order - 1) / 3.0f);
        
        return ComplexFloat(i_symbol / normalization, q_symbol / normalization);
    }
};

/// Generate FFT test data
class FFTDataGenerator {
private:
    RandomGenerator rng_;
    
public:
    explicit FFTDataGenerator(uint32_t seed = 42) : rng_(seed) {}
    
    /// Generate sinusoidal test signal
    std::vector<ComplexFloat> generate_sinusoid(
        size_t num_samples,
        float frequency,
        float sample_rate,
        float amplitude = 1.0f,
        float phase = 0.0f) const {
        
        std::vector<ComplexFloat> signal;
        signal.reserve(num_samples);
        
        float angular_freq = 2.0f * M_PI * frequency / sample_rate;
        
        for (size_t n = 0; n < num_samples; ++n) {
            float t = static_cast<float>(n);
            float real_part = amplitude * std::cos(angular_freq * t + phase);
            float imag_part = amplitude * std::sin(angular_freq * t + phase);
            signal.emplace_back(real_part, imag_part);
        }
        
        return signal;
    }
    
    /// Generate multi-tone signal
    std::vector<ComplexFloat> generate_multitone(
        size_t num_samples,
        const std::vector<float>& frequencies,
        float sample_rate,
        const std::vector<float>& amplitudes = {},
        const std::vector<float>& phases = {}) const {
        
        std::vector<ComplexFloat> signal(num_samples, ComplexFloat(0.0f, 0.0f));
        
        for (size_t tone = 0; tone < frequencies.size(); ++tone) {
            float amplitude = (amplitudes.empty()) ? 1.0f : amplitudes[tone];
            float phase = (phases.empty()) ? 0.0f : phases[tone];
            
            auto tone_signal = generate_sinusoid(num_samples, frequencies[tone], 
                                               sample_rate, amplitude, phase);
            
            for (size_t n = 0; n < num_samples; ++n) {
                signal[n] += tone_signal[n];
            }
        }
        
        return signal;
    }
    
    /// Generate OFDM-like signal
    std::vector<ComplexFloat> generate_ofdm_symbol(
        size_t fft_size,
        size_t num_used_subcarriers,
        const std::vector<ComplexFloat>& data_symbols = {}) const {
        
        std::vector<ComplexFloat> frequency_domain(fft_size, ComplexFloat(0.0f, 0.0f));
        
        // DC and edge subcarriers are typically unused in OFDM
        size_t guard_subcarriers = (fft_size - num_used_subcarriers) / 2;
        
        for (size_t i = 0; i < num_used_subcarriers; ++i) {
            size_t subcarrier_index = guard_subcarriers + i;
            
            if (!data_symbols.empty() && i < data_symbols.size()) {
                frequency_domain[subcarrier_index] = data_symbols[i];
            } else {
                // Generate random QAM symbol if no data provided
                frequency_domain[subcarrier_index] = rng_.complex_normal(0.0f, 1.0f);
            }
        }
        
        return frequency_domain;
    }
    
    /// Generate chirp signal for frequency domain testing
    std::vector<ComplexFloat> generate_chirp(
        size_t num_samples,
        float start_freq,
        float end_freq,
        float sample_rate,
        float amplitude = 1.0f) const {
        
        std::vector<ComplexFloat> signal;
        signal.reserve(num_samples);
        
        float freq_slope = (end_freq - start_freq) / (num_samples - 1);
        
        for (size_t n = 0; n < num_samples; ++n) {
            float t = static_cast<float>(n) / sample_rate;
            float instantaneous_freq = start_freq + freq_slope * n;
            float phase = 2.0f * M_PI * (start_freq * t + 0.5f * freq_slope * t * t);
            
            float real_part = amplitude * std::cos(phase);
            float imag_part = amplitude * std::sin(phase);
            signal.emplace_back(real_part, imag_part);
        }
        
        return signal;
    }
};

/// Generate MIMO system test data
class MIMODataGenerator {
private:
    RandomGenerator rng_;
    
public:
    explicit MIMODataGenerator(uint32_t seed = 42) : rng_(seed) {}
    
    /// Generate MIMO channel matrix
    std::vector<std::vector<ComplexFloat>> generate_mimo_channel(
        size_t num_tx_antennas,
        size_t num_rx_antennas,
        float correlation = 0.0f) const {
        
        std::vector<std::vector<ComplexFloat>> channel(
            num_rx_antennas, std::vector<ComplexFloat>(num_tx_antennas)
        );
        
        for (size_t rx = 0; rx < num_rx_antennas; ++rx) {
            for (size_t tx = 0; tx < num_tx_antennas; ++tx) {
                // Independent Rayleigh fading for each antenna pair
                channel[rx][tx] = rng_.complex_normal(0.0f, 1.0f / std::sqrt(2.0f));
                
                // Apply spatial correlation if specified
                if (correlation > 0.0f && tx > 0) {
                    ComplexFloat correlated_component = 
                        channel[rx][tx-1] * correlation + 
                        rng_.complex_normal(0.0f, std::sqrt(1.0f - correlation * correlation));
                    channel[rx][tx] = correlated_component;
                }
            }
        }
        
        return channel;
    }
    
    /// Generate transmitted symbol matrix
    std::vector<std::vector<ComplexFloat>> generate_mimo_symbols(
        size_t num_tx_antennas,
        size_t num_symbols,
        int modulation_order = 16) const {
        
        std::vector<std::vector<ComplexFloat>> symbols(
            num_tx_antennas, std::vector<ComplexFloat>(num_symbols)
        );
        
        ModulationDataGenerator mod_gen(rng_.uniform() * 10000);
        
        for (size_t tx = 0; tx < num_tx_antennas; ++tx) {
            auto bits = mod_gen.generate_bits(num_symbols * static_cast<int>(std::log2(modulation_order)));
            auto qam_symbols = mod_gen.generate_qam_symbols(bits, modulation_order);
            
            for (size_t sym = 0; sym < num_symbols; ++sym) {
                symbols[tx][sym] = qam_symbols[sym];
            }
        }
        
        return symbols;
    }
    
    /// Generate received symbols with MIMO channel and noise
    std::vector<std::vector<ComplexFloat>> generate_mimo_received(
        const std::vector<std::vector<ComplexFloat>>& transmitted_symbols,
        const std::vector<std::vector<ComplexFloat>>& channel_matrix,
        float snr_db) const {
        
        size_t num_rx_antennas = channel_matrix.size();
        size_t num_tx_antennas = channel_matrix[0].size();
        size_t num_symbols = transmitted_symbols[0].size();
        
        assert(transmitted_symbols.size() == num_tx_antennas);
        
        std::vector<std::vector<ComplexFloat>> received(
            num_rx_antennas, std::vector<ComplexFloat>(num_symbols, ComplexFloat(0.0f, 0.0f))
        );
        
        float noise_power = std::pow(10.0f, -snr_db / 10.0f);
        float noise_std = std::sqrt(noise_power / 2.0f);
        
        for (size_t sym = 0; sym < num_symbols; ++sym) {
            for (size_t rx = 0; rx < num_rx_antennas; ++rx) {
                // Channel multiplication
                for (size_t tx = 0; tx < num_tx_antennas; ++tx) {
                    received[rx][sym] += channel_matrix[rx][tx] * transmitted_symbols[tx][sym];
                }
                
                // Add noise
                ComplexFloat noise = rng_.complex_normal(0.0f, noise_std);
                received[rx][sym] += noise;
            }
        }
        
        return received;
    }
};

/// Utility functions for data validation
namespace validation {

/// Check if complex vector has reasonable magnitude
inline bool validate_complex_magnitude(const std::vector<ComplexFloat>& data, 
                                       float max_magnitude = 10.0f) {
    for (const auto& sample : data) {
        if (std::abs(sample) > max_magnitude) {
            return false;
        }
    }
    return true;
}

/// Calculate average power of complex signal
inline float calculate_average_power(const std::vector<ComplexFloat>& signal) {
    if (signal.empty()) return 0.0f;
    
    float total_power = 0.0f;
    for (const auto& sample : signal) {
        total_power += std::norm(sample);
    }
    
    return total_power / signal.size();
}

/// Verify that generated data has expected statistical properties
inline bool validate_noise_statistics(const std::vector<ComplexFloat>& noise_samples,
                                      float expected_power,
                                      float tolerance = 0.1f) {
    float measured_power = calculate_average_power(noise_samples);
    return std::abs(measured_power - expected_power) <= tolerance * expected_power;
}

} // namespace validation

} // namespace data_generators