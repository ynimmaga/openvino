// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"
#include "runner.hpp"

namespace ov {
namespace ggml_plugin {

using ov::ggml_cgraph_loader::GgmlModel;

/// dump_vision artifact (vision tower + projector, one tile) as an OpenVINO vision embeddings
/// model: pixel_values [n_tiles, 3, px, px] -> image_features [n_tiles, n_image_tokens, hidden].
/// The graph is shape-static for one tile, so tiles run one after another.
class VisionRunner : public Runner {
public:
    explicit VisionRunner(std::shared_ptr<GgmlModel> model);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::string backend_name() const override;

private:
    std::shared_ptr<GgmlModel> m_model;
    size_t m_px = 0, m_hidden = 0, m_n_tokens = 0;
};

/// dump_cgraph DUMP_TOKEN_EMBD artifact as an OpenVINO text embeddings model:
/// input_ids [1, seq] -> inputs_embeds [1, seq, hidden], one token per graph run.
class TokenEmbeddingsRunner : public Runner {
public:
    explicit TokenEmbeddingsRunner(std::shared_ptr<GgmlModel> model);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::string backend_name() const override;

private:
    std::shared_ptr<GgmlModel> m_model;
    size_t m_hidden = 0;
    size_t m_rows = 0;  // embedding table rows (vocabulary size); 0 if unknown
};

}  // namespace ggml_plugin
}  // namespace ov
