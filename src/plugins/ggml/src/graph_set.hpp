// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
};

}  // namespace ggml_plugin
}  // namespace ov
