// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace ggml_plugin {

/// A KV cache held in ggml backend buffers. Its data is not copied out; the state is exposed as its
/// sequence length only: get_state() returns a [1, 1, len, 1] tensor (contents unspecified) and
/// set_state() with a shorter one truncates the cache -- what GenAI does to trim KV in chat mode.
/// Slots past the length are masked out by the runner, so neither reset nor truncation touches
/// the device buffers.
class SequenceState : public ov::IVariableState {
public:
    SequenceState(const std::string& name, size_t& length) : ov::IVariableState(name), m_length(length) {}

    void reset() override {
        m_length = 0;
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        const auto& shape = state->get_shape();
        OPENVINO_ASSERT(shape.size() == 4, "[GGML] KV cache state is [1, 1, seq_len, 1], got rank ", shape.size());
        OPENVINO_ASSERT(shape[2] <= m_length,
                        "[GGML] KV cache state can only be truncated (have ",
                        m_length,
                        " tokens, got ",
                        shape[2],
                        ")");
        m_length = shape[2];
    }

    ov::SoPtr<ov::ITensor> get_state() const override {
        return ov::make_tensor(ov::element::u8, ov::Shape{1, 1, m_length, 1});
    }

private:
    size_t& m_length;
};

}  // namespace ggml_plugin
}  // namespace ov
