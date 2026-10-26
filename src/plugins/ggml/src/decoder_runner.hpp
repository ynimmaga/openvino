// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_set.hpp"

namespace ov {
namespace ggml_plugin {

using ov::ggml_cgraph_loader::GgmlModel;

class DecoderRunner : public Runner {
public:
    DecoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::vector<ov::SoPtr<ov::IVariableState>> states() const override;
    std::string backend_name() const override;

private:
    struct Rows {
        size_t first_index = 0;  // sequence index of the first row; rows are consecutive
        std::vector<int32_t> positions;
        const int64_t* tokens = nullptr;
        const float* embeds = nullptr;
        const int64_t* mask = nullptr;  // attention_mask over the sequence
    };

    void forward(const Rows& rows, const GraphSet::Graph& graph);

    GraphSet m_graphs;
    std::shared_ptr<GgmlModel> m_active;  // graph that ran last (holds the logits)
    bool m_embeds = false;
    size_t m_hidden = 0;
    size_t m_vocab = 0;
    size_t m_scratch_slots = 0;
    size_t m_cached = 0;  // tokens currently held in the KV cache
    // Recurrent layers (Mamba / RWKV / linear attention) fold every token into a state that cannot
    // be rewound: a new sequence zeroes it, and it never sees padding rows.
    bool m_recurrent = false;
    // Qwen3.8-Flash-Next hashes each token with its predecessors into per-layer embedding rows (n-gram
    // "PLE"); the graph takes the row indices as an input that only the host can compute.
    struct Ple {
        int64_t ngram = 0, per_gram = 0, n_heads = 0, eos = 0;
        std::vector<uint64_t> multipliers, offsets, vocab;
        std::string input;  // the unnamed graph input holding [n_heads, rows] indices
    } m_ple;
    std::vector<int64_t> m_history;  // the sequence's token ids so far (n-gram contexts)
    ov::SoPtr<ov::IVariableState> m_state;
};

}  // namespace ggml_plugin
}  // namespace ov
