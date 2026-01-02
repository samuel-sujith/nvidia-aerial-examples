/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Framework stubs to resolve linking issues
// These provide minimal implementations of framework functions that examples depend on

namespace framework {
namespace log {

// Stub implementation for logger default level
int get_logger_default_level() {
    return 2; // INFO level
}

namespace detail {
// Stub implementation for Quill logger
void* get_quill_logger() {
    return nullptr; // Return null pointer for stub
}
} // namespace detail

} // namespace log
} // namespace framework