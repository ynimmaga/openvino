// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_set.hpp"

namespace ov {
namespace ggml_plugin {

/// The Qwen-Image 2.1 diffusion transformer, as captured from stable-diffusion.cpp (one artifact per text
/// length and latent grid, selected by the inputs' shapes). It takes the inputs OpenVINO GenAI's
/// QwenImage21Transformer2DModel feeds its exported graph (hidden_states, encoder_hidden_states, timestep, cos, sin,
/// gather_idx, attn_mask, modulation_mask) and returns out_hidden_states over the joint sequence.
///
/// Only the text-to-image layout is supported: causal text followed by one fully-attending image block. The graph
/// builds that structure itself, so the mask / gather / modulation inputs are checked, not forwarded.
class DitRunner : public Runner {
public:
    DitRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::string backend_name() const override;

private:
    struct Graph {
        size_t text = 0;    // prompt tokens
        size_t width = 0;   // latent grid
        size_t height = 0;
        std::shared_ptr<GgmlModel> model;
    };
    static Graph describe(std::shared_ptr<GgmlModel> model);

    std::vector<Graph> m_graphs;
    size_t m_channels = 0;  // latent channels per token
    size_t m_context = 0;   // text embedding width
    size_t m_head_dim = 0;
};

}  // namespace ggml_plugin
}  // namespace ov
