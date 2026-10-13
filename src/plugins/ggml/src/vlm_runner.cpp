// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vlm_runner.hpp"

#include <algorithm>
#include <cmath>

#include "openvino/core/except.hpp"

namespace ov {
namespace ggml_plugin {

// -------------------------------------------------------------------------------------------- vision

VisionRunner::VisionRunner(std::shared_ptr<GgmlModel> model) : m_model(std::move(model)) {
    // inp_raw is [px, px, 3] in ggml order, i.e. one CHW tile.
    m_px = static_cast<size_t>(std::lround(std::sqrt(m_model->input_size("inp_raw") / 3.0)));
    OPENVINO_ASSERT(m_px * m_px * 3 == m_model->input_size("inp_raw"), "[GGML] vision input is not a square RGB tile");
    m_hidden = m_model->logits_size();
    m_n_tokens = m_model->output_size() / m_hidden;
}

std::vector<PortDesc> VisionRunner::inputs() const {
    const auto px = static_cast<int64_t>(m_px);
    return {{"pixel_values", ov::element::f32, ov::PartialShape{-1, 3, px, px}}};
}

std::vector<PortDesc> VisionRunner::outputs() const {
    return {{"image_features",
             ov::element::f32,
             ov::PartialShape{-1, static_cast<int64_t>(m_n_tokens), static_cast<int64_t>(m_hidden)}}};
}

void VisionRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const auto& pixels = inputs.at("pixel_values");
    const auto shape = pixels->get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && shape[1] == 3 && shape[2] == m_px && shape[3] == m_px,
                    "[GGML] vision artifact was dumped for [N, 3, ",
                    m_px,
                    ", ",
                    m_px,
                    "] tiles, got ",
                    shape);
    const size_t n_tiles = shape[0];
    const size_t tile = 3 * m_px * m_px;
    auto& features = outputs.at("image_features");
    features->set_shape({n_tiles, m_n_tokens, m_hidden});
    for (size_t i = 0; i < n_tiles; i++) {
        m_model->write_input("inp_raw", static_cast<const float*>(pixels->data()) + i * tile, tile * sizeof(float));
        OPENVINO_ASSERT(m_model->compute(), "[GGML] vision compute failed on tile ", i);
        m_model->read_output(static_cast<float*>(features->data()) + i * m_n_tokens * m_hidden);
    }
}

std::string VisionRunner::backend_name() const {
    return m_model->backend_name();
}

// ---------------------------------------------------------------------------------- token embeddings

TokenEmbeddingsRunner::TokenEmbeddingsRunner(std::shared_ptr<GgmlModel> model) : m_model(std::move(model)) {
    OPENVINO_ASSERT(m_model->input_size("inp_tokens") == 1, "[GGML] token embedding artifact must take one token");
    m_hidden = m_model->output_size();
    std::vector<std::string> vocab;
    if (m_model->gguf_meta_str_array("tokenizer.ggml.tokens", vocab)) {
        m_rows = vocab.size();
    }
}

std::vector<PortDesc> TokenEmbeddingsRunner::inputs() const {
    return {{"input_ids", ov::element::i64, ov::PartialShape{-1, -1}}};
}

std::vector<PortDesc> TokenEmbeddingsRunner::outputs() const {
    return {{"inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, static_cast<int64_t>(m_hidden)}}};
}

void TokenEmbeddingsRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const auto& ids = inputs.at("input_ids");
    const auto shape = ids->get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "[GGML] input_ids must be [batch, seq]");
    auto& embeds = outputs.at("inputs_embeds");
    embeds->set_shape({shape[0], shape[1], m_hidden});
    const auto* tokens = static_cast<const int64_t*>(ids->data());
    auto* out = static_cast<float*>(embeds->data());
    for (size_t i = 0; i < ids->get_size(); i++) {
        // A tokenizer can add placeholder tokens the .gguf has no embedding for (llava-hf's
        // <image>, <pad>); the pipeline replaces those rows with image features, so they get zeros.
        if (m_rows && static_cast<size_t>(tokens[i]) >= m_rows) {
            std::fill(out + i * m_hidden, out + (i + 1) * m_hidden, 0.0f);
            continue;
        }
        const int32_t token = static_cast<int32_t>(tokens[i]);
        m_model->write_input("inp_tokens", &token, sizeof(token));
        OPENVINO_ASSERT(m_model->compute(), "[GGML] token embedding compute failed");
        m_model->read_output(out + i * m_hidden);
    }
}

std::string TokenEmbeddingsRunner::backend_name() const {
    return m_model->backend_name();
}

}  // namespace ggml_plugin
}  // namespace ov
