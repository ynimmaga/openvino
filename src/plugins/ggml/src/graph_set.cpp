// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_set.hpp"

#include <algorithm>
#include <set>

#include "openvino/core/except.hpp"

namespace ov {
namespace ggml_plugin {

size_t GraphSet::rows_of(const GgmlModel& model) {
    const size_t tokens = model.input_size("inp_tokens");
    return tokens ? tokens : model.input_size("inp_pos");
}

size_t GraphSet::positions_per_token(const GgmlModel& model) {
    const size_t rows = rows_of(model), pos = model.input_size("inp_pos");
    return rows && pos ? pos / rows : 0;
}

GraphSet::GraphSet(std::shared_ptr<GgmlModel> main, const RunnerConfig& config) {
    const size_t main_tokens = rows_of(*main);
    const size_t main_n_kv = main->context_size();
    OPENVINO_ASSERT(main_n_kv > 0, "[GGML] decoder artifact reports no context size");
    m_graphs.push_back({main_tokens, main_n_kv, std::move(main)});
    for (const auto& path : config.buckets) {
        auto model = GgmlModel::from_cgraph(path.string(), m_graphs.front().model);
        Graph g{rows_of(*model), model->context_size(), model};
        OPENVINO_ASSERT(g.n_kv <= main_n_kv,
                        "[GGML] ",
                        path,
                        " has n_kv ",
                        g.n_kv,
                        " > the main artifact's ",
                        main_n_kv,
                        ": the main artifact must be the largest window (it owns the KV cache)");
        OPENVINO_ASSERT(model->logits_size() == m_graphs.front().model->logits_size(),
                        "[GGML] all graphs must share one vocabulary");
        m_graphs.push_back(std::move(g));
    }
}

size_t GraphSet::max_tokens() const {
    size_t m = 0;
    for (const auto& g : m_graphs) {
        m = std::max(m, g.tokens);
    }
    return m;
}

const GraphSet::Graph& GraphSet::pick(size_t remaining,
                                      const std::function<size_t(size_t rows)>& slots_needed) const {
    std::set<size_t> token_counts;
    for (const auto& g : m_graphs) {
        token_counts.insert(g.tokens);
    }
    auto t = token_counts.lower_bound(remaining);
    const size_t preferred = t != token_counts.end() ? *t : *token_counts.rbegin();
    // A token count may only be dumped for small windows; fall back to fewer rows per pass then.
    size_t need = 0;
    for (auto it = token_counts.rbegin(); it != token_counts.rend(); ++it) {
        if (*it > preferred) {
            continue;
        }
        need = slots_needed(std::min(remaining, *it));
        const Graph* best = nullptr;
        for (const auto& g : m_graphs) {
            if (g.tokens == *it && g.n_kv >= need && (!best || g.n_kv < best->n_kv)) {
                best = &g;
            }
        }
        if (best) {
            return *best;
        }
    }
    OPENVINO_THROW("[GGML] ", need, " KV slots are needed but no graph covers them; re-dump with a larger n_kv");
}

}  // namespace ggml_plugin
}  // namespace ov
