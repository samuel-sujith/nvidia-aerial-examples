#pragma once

#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>

namespace test_utils {

/// Complex number type alias
using ComplexFloat = std::complex<float>;

/// Test result structure
struct TestResult {
    bool passed{false};
    std::string description;
    double actual_value{0.0};
    double expected_value{0.0};
    double tolerance{1e-6};
    
    TestResult(bool p, const std::string& desc) 
        : passed(p), description(desc) {}
    
    TestResult(bool p, const std::string& desc, double actual, double expected, double tol = 1e-6)
        : passed(p), description(desc), actual_value(actual), expected_value(expected), tolerance(tol) {}
};

/// Test suite for collecting multiple test results
class TestSuite {
private:
    std::vector<TestResult> results_;
    std::string suite_name_;
    
public:
    explicit TestSuite(const std::string& name) : suite_name_(name) {}
    
    void add_result(const TestResult& result) {
        results_.push_back(result);
    }
    
    void add_test(bool passed, const std::string& description) {
        results_.emplace_back(passed, description);
    }
    
    void add_numerical_test(const std::string& description, double actual, double expected, double tolerance = 1e-6) {
        bool passed = std::abs(actual - expected) <= tolerance;
        results_.emplace_back(passed, description, actual, expected, tolerance);
    }
    
    bool all_passed() const {
        return std::all_of(results_.begin(), results_.end(), 
                          [](const TestResult& r) { return r.passed; });
    }
    
    void print_summary() const {
        std::cout << "=== Test Suite: " << suite_name_ << " ===\n";
        
        size_t passed = 0;
        for (const auto& result : results_) {
            std::cout << "[" << (result.passed ? "PASS" : "FAIL") << "] " 
                     << result.description;
            
            if (!result.passed && result.actual_value != 0.0) {
                std::cout << " (Expected: " << result.expected_value 
                         << ", Actual: " << result.actual_value 
                         << ", Tolerance: " << result.tolerance << ")";
            }
            std::cout << "\n";
            
            if (result.passed) passed++;
        }
        
        std::cout << "\nResults: " << passed << "/" << results_.size() << " tests passed\n";
        std::cout << "Success Rate: " << (100.0 * passed / results_.size()) << "%\n\n";
    }
    
    size_t num_tests() const { return results_.size(); }
    size_t num_passed() const {
        return std::count_if(results_.begin(), results_.end(), 
                           [](const TestResult& r) { return r.passed; });
    }
};

/// Floating point comparison with tolerance
inline bool float_equals(float a, float b, float tolerance = 1e-6f) {
    return std::abs(a - b) <= tolerance;
}

inline bool double_equals(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance;
}

/// Complex number comparison
inline bool complex_equals(const ComplexFloat& a, const ComplexFloat& b, float tolerance = 1e-6f) {
    return float_equals(a.real(), b.real(), tolerance) && 
           float_equals(a.imag(), b.imag(), tolerance);
}

/// Vector comparison utilities
template<typename T>
bool vectors_equal(const std::vector<T>& a, const std::vector<T>& b, double tolerance = 1e-6) {
    if (a.size() != b.size()) return false;
    
    for (size_t i = 0; i < a.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!double_equals(static_cast<double>(a[i]), static_cast<double>(b[i]), tolerance)) {
                return false;
            }
        } else {
            if (a[i] != b[i]) return false;
        }
    }
    return true;
}

/// Complex vector comparison
inline bool complex_vectors_equal(const std::vector<ComplexFloat>& a, 
                                  const std::vector<ComplexFloat>& b, 
                                  float tolerance = 1e-6f) {
    if (a.size() != b.size()) return false;
    
    for (size_t i = 0; i < a.size(); ++i) {
        if (!complex_equals(a[i], b[i], tolerance)) {
            return false;
        }
    }
    return true;
}

/// Calculate Mean Squared Error between two vectors
template<typename T>
double calculate_mse(const std::vector<T>& actual, const std::vector<T>& expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        sum += diff * diff;
    }
    
    return sum / actual.size();
}

/// Calculate Root Mean Squared Error
template<typename T>
double calculate_rmse(const std::vector<T>& actual, const std::vector<T>& expected) {
    return std::sqrt(calculate_mse(actual, expected));
}

/// Calculate Signal-to-Noise Ratio in dB
template<typename T>
double calculate_snr_db(const std::vector<T>& signal, const std::vector<T>& noise) {
    if (signal.size() != noise.size()) {
        throw std::invalid_argument("Signal and noise vectors must have the same size");
    }
    
    double signal_power = 0.0;
    double noise_power = 0.0;
    
    for (size_t i = 0; i < signal.size(); ++i) {
        double s = static_cast<double>(signal[i]);
        double n = static_cast<double>(noise[i]);
        signal_power += s * s;
        noise_power += n * n;
    }
    
    if (noise_power == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    return 10.0 * std::log10(signal_power / noise_power);
}

/// Test data validation utilities
class DataValidator {
public:
    /// Check if data is finite (no NaN or infinity)
    template<typename T>
    static bool is_finite_data(const std::vector<T>& data) {
        return std::all_of(data.begin(), data.end(), [](T val) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isfinite(val);
            } else {
                return true; // Non-floating point types are always finite
            }
        });
    }
    
    /// Check if complex data is finite
    static bool is_finite_complex_data(const std::vector<ComplexFloat>& data) {
        return std::all_of(data.begin(), data.end(), [](const ComplexFloat& val) {
            return std::isfinite(val.real()) && std::isfinite(val.imag());
        });
    }
    
    /// Check if data is within expected range
    template<typename T>
    static bool is_in_range(const std::vector<T>& data, T min_val, T max_val) {
        return std::all_of(data.begin(), data.end(), [min_val, max_val](T val) {
            return val >= min_val && val <= max_val;
        });
    }
    
    /// Check if complex data magnitude is within range
    static bool is_complex_magnitude_in_range(const std::vector<ComplexFloat>& data, 
                                               float min_mag, float max_mag) {
        return std::all_of(data.begin(), data.end(), [min_mag, max_mag](const ComplexFloat& val) {
            float mag = std::abs(val);
            return mag >= min_mag && mag <= max_mag;
        });
    }
};

/// Memory alignment testing utilities
class AlignmentTester {
public:
    /// Check if pointer is aligned to specified boundary
    static bool is_aligned(void* ptr, size_t alignment) {
        return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
    }
    
    /// Check if vector data is aligned
    template<typename T>
    static bool is_vector_aligned(const std::vector<T>& vec, size_t alignment) {
        return is_aligned(const_cast<void*>(static_cast<const void*>(vec.data())), alignment);
    }
    
    /// Get pointer alignment
    static size_t get_alignment(void* ptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr == 0) return SIZE_MAX; // Null pointer
        
        size_t alignment = 1;
        while ((addr % (alignment * 2)) == 0) {
            alignment *= 2;
        }
        return alignment;
    }
};

/// Print utilities for debugging
namespace print {

template<typename T>
void vector(const std::vector<T>& data, const std::string& name = "", size_t max_elements = 10) {
    std::cout << name << " [size=" << data.size() << "]: ";
    
    size_t elements_to_print = std::min(data.size(), max_elements);
    for (size_t i = 0; i < elements_to_print; ++i) {
        std::cout << data[i];
        if (i < elements_to_print - 1) std::cout << ", ";
    }
    
    if (data.size() > max_elements) {
        std::cout << ", ...";
    }
    std::cout << "\n";
}

inline void complex_vector(const std::vector<ComplexFloat>& data, 
                          const std::string& name = "", 
                          size_t max_elements = 10) {
    std::cout << name << " [size=" << data.size() << "]: ";
    
    size_t elements_to_print = std::min(data.size(), max_elements);
    for (size_t i = 0; i < elements_to_print; ++i) {
        std::cout << "(" << data[i].real() << "+" << data[i].imag() << "j)";
        if (i < elements_to_print - 1) std::cout << ", ";
    }
    
    if (data.size() > max_elements) {
        std::cout << ", ...";
    }
    std::cout << "\n";
}

} // namespace print

} // namespace test_utils