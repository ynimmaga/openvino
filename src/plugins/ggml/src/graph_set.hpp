// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"
#include "runner.hpp"

namespace ov {
namespace ggml_plugin {

using ov::ggml_cgraph_loader::GgmlModel;

/// The graphs of one decoder: the main artifact plus GGML_BUCKETS, each dumped for some token count
/// (N_TOKENS) and context size (n_kv). They share the main artifact's weights and KV cache, so the
/// main artifact must have the largest n_kv; smaller ones see a view of the cache's first slots.
/// Attention spans a graph's whole window, so running the smallest window that covers a sequence
/// keeps short sequences fast while long ones still fit.
/// IEEE half from float, for mask values (small integers, -inf); no denormals.
inline uint16_t to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t e = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    const uint32_t m = x & 0x7FFFFFu;
    if (e <= 0) return static_cast<uint16_t>(sign);
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    uint32_t h = (static_cast<uint32_t>(e) << 10) | (m >> 13);
    if ((m & 0x1FFFu) > 0x1000u || ((m & 0x1FFFu) == 0x1000u && (h & 1u))) h++;
    return static_cast<uint16_t>(sign | h);
}

class GraphSet {
public:
    struct Graph {
        size_t tokens = 0;  // rows per forward pass
        size_t n_kv = 0;    // KV slots the graph attends over
        std::shared_ptr<GgmlModel> model;
    };

    GraphSet(std::shared_ptr<GgmlModel> main, const RunnerConfig& config);

    /// Rows (tokens) per forward pass of a decoder graph.
    static size_t rows_of(const GgmlModel& model);
    /// RoPE positions per token: 1, 4 with M-RoPE (stored section-major), 0 without RoPE (ALiBi).
    static size_t positions_per_token(const GgmlModel& model);

    const GgmlModel& main() const {
        return *m_graphs.front().model;
    }
    GgmlModel& main() {
        return *m_graphs.front().model;
    }
    size_t max_tokens() const;

    /// Fills the graph's "inp_attn_scale" input, if it has one: llama.cpp's position-dependent attention
    /// temperature, log(floor((pos + offset) / floor_scale) + 1) * scale + 1 (Mistral-3, Llama-4, DeepSeek-2).
    /// `positions` are the rows' RoPE positions; padding rows get 1.
    void write_attn_scale(GgmlModel& model, const std::vector<int32_t>& positions) const;
    size_t max_n_kv() const {
        return m_graphs.front().n_kv;
    }
    const std::vector<Graph>& graphs() const {
        return m_graphs;
    }

    /// The graph for the next forward pass over `remaining` rows: the smallest token count that fits
    /// them (else the largest), then the smallest window holding the slots that `slots_needed(rows)`
    /// reports for that many rows. Throws when no window is large enough.
    const Graph& pick(size_t remaining, const std::function<size_t(size_t rows)>& slots_needed) const;

private:
    std::vector<Graph> m_graphs;  // [0] is the main artifact
    float m_temp_scale = 0.0f, m_temp_offset = 0.0f;
    int32_t m_temp_floor = 0;
};

}  // namespace ggml_plugin
}  // namespace ov
