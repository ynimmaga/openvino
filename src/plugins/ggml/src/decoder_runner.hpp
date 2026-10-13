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
    ov::SoPtr<ov::IVariableState> m_state;
};

}  // namespace ggml_plugin
}  // namespace ov
