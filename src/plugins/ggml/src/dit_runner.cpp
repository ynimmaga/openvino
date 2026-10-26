// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dit_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "openvino/core/except.hpp"

namespace ov {
namespace ggml_plugin {

namespace {

constexpr size_t CHANNELS = 64;    // packed latent channels per token (16 VAE channels x 2x2 patch)
constexpr size_t CONTEXT = 4096;   // Qwen3-VL hidden size
constexpr size_t HEAD_DIM = 128;   // attention head width; the rotary tables carry HEAD_DIM / 2 angles per token

}  // namespace

DitRunner::Graph DitRunner::describe(std::shared_ptr<GgmlModel> model) {
    Graph g;
    const size_t mask = model->input_size("dit_mask_0");
    g.text = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(mask))));
    OPENVINO_ASSERT(g.text * g.text == mask && g.text > 0, "[GGML] DiT artifact: the text mask is not square");
    OPENVINO_ASSERT(model->input_size("dit_context") == g.text * CONTEXT,
                    "[GGML] DiT artifact: the context input is not [",
                    g.text,
                    ", ",
                    CONTEXT,
                    "]");
    const size_t latent = model->input_size("dit_latent");
    OPENVINO_ASSERT(latent % CHANNELS == 0, "[GGML] DiT artifact: the latent is not ", CHANNELS, "-channel");
    g.width = latent / CHANNELS;  // image tokens; the grid shape does not matter to the runner
    g.height = 1;
    OPENVINO_ASSERT(model->input_size("dit_pe") == 4 * (HEAD_DIM / 2) * (g.text + g.width),
                    "[GGML] DiT artifact: the rotary table does not cover ",
                    g.text,
                    " text + ",
                    g.width,
                    " image tokens");
    g.model = std::move(model);
    return g;
}

DitRunner::DitRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config) {
    m_graphs.push_back(describe(model));
    for (const auto& path : config.buckets) {
        m_graphs.push_back(describe(GgmlModel::from_cgraph(path.string(), model)));
    }
    m_channels = CHANNELS;
    m_context = CONTEXT;
    m_head_dim = HEAD_DIM;
}

std::vector<PortDesc> DitRunner::inputs() const {
    const auto head = static_cast<int64_t>(m_head_dim);
    return {{"hidden_states", ov::element::f32, ov::PartialShape{-1, -1, -1}},
            {"encoder_hidden_states", ov::element::f32, ov::PartialShape{-1, -1, -1}},
            {"timestep", ov::element::f32, ov::PartialShape{-1}},
            {"cos", ov::element::f32, ov::PartialShape{1, -1, 1, head}},
            {"sin", ov::element::f32, ov::PartialShape{1, -1, 1, head}},
            {"gather_idx", ov::element::i64, ov::PartialShape{-1}},
            {"attn_mask", ov::element::f32, ov::PartialShape{-1, 1, -1, -1}},
            {"modulation_mask", ov::element::boolean, ov::PartialShape{-1}}};
}

std::vector<PortDesc> DitRunner::outputs() const {
    return {{"out_hidden_states", ov::element::f32, ov::PartialShape{-1, -1, static_cast<int64_t>(m_channels)}}};
}

std::string DitRunner::backend_name() const {
    return m_graphs.front().model->backend_name();
}

void DitRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const auto hs_shape = inputs.at("hidden_states")->get_shape();
    const auto ctx_shape = inputs.at("encoder_hidden_states")->get_shape();
    OPENVINO_ASSERT(hs_shape.size() == 3 && ctx_shape.size() == 3 && hs_shape[2] == m_channels &&
                        ctx_shape[2] == m_context && hs_shape[0] == ctx_shape[0],
                    "[GGML] DiT: unexpected hidden_states / encoder_hidden_states shapes");
    const size_t batch = hs_shape[0], image = hs_shape[1], text = ctx_shape[1], joint = text + image;

    const auto it = std::find_if(m_graphs.begin(), m_graphs.end(), [&](const Graph& g) {
        return g.text == text && g.width == image;
    });
    if (it == m_graphs.end()) {
        std::string have;
        for (const auto& g : m_graphs) {
            have += " (" + std::to_string(g.text) + " text, " + std::to_string(g.width) + " image tokens)";
        }
        OPENVINO_THROW("[GGML] no DiT graph for ",
                       text,
                       " text and ",
                       image,
                       " image tokens; dumped for:",
                       have,
                       ". Capture another with stable-diffusion.cpp for this prompt length and image size");
    }
    auto& model = *it->model;

    // The graph builds the text-to-image joint sequence itself; refuse anything else (image editing, several images).
    const auto* gather = inputs.at("gather_idx")->data<int64_t>();
    OPENVINO_ASSERT(inputs.at("gather_idx")->get_size() == joint, "[GGML] DiT: gather_idx does not span the joint sequence");
    for (size_t i = 0; i < joint; i++) {
        OPENVINO_ASSERT(gather[i] == static_cast<int64_t>(i), "[GGML] DiT: only the text-to-image joint layout is supported");
    }
    const auto* mod = inputs.at("modulation_mask")->data<char>();
    for (size_t i = 0; i < joint; i++) {
        OPENVINO_ASSERT((mod[i] != 0) == (i >= text), "[GGML] DiT: only the text-to-image modulation layout is supported");
    }
    const auto* attn = inputs.at("attn_mask")->data<float>();
    for (size_t q = 0; q < joint; q++) {
        for (size_t k = 0; k < joint; k++) {
            const bool allowed = k <= q || (q >= text && k >= text);
            OPENVINO_ASSERT((attn[q * joint + k] == 0.0f) == allowed,
                            "[GGML] DiT: only the text-to-image attention layout is supported");
        }
    }

    // Rotary angles -> stable-diffusion.cpp's per-token 2x2 matrices [[cos, -sin], [sin, cos]], one per angle.
    const size_t half = m_head_dim / 2;
    const auto* cos_in = inputs.at("cos")->data<float>();
    const auto* sin_in = inputs.at("sin")->data<float>();
    OPENVINO_ASSERT(inputs.at("cos")->get_size() == joint * m_head_dim, "[GGML] DiT: rotary tables do not span the joint sequence");
    std::vector<float> pe(4 * half * joint);
    for (size_t t = 0; t < joint; t++) {
        for (size_t k = 0; k < half; k++) {
            const float c = cos_in[t * m_head_dim + k], s = sin_in[t * m_head_dim + k];
            float* m = &pe[(t * half + k) * 4];
            m[0] = c;   // (0, 0)
            m[1] = -s;  // (1, 0)
            m[2] = s;   // (0, 1)
            m[3] = c;   // (1, 1)
        }
    }
    model.write_input("dit_pe", pe.data(), pe.size() * sizeof(float));

    std::vector<float> text_mask(text * text);
    for (size_t q = 0; q < text; q++) {
        for (size_t k = 0; k < text; k++) {
            text_mask[q * text + k] = k <= q ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    model.write_input("dit_mask_0", text_mask.data(), text_mask.size() * sizeof(float));

    auto& out = outputs.at("out_hidden_states");
    out->set_shape({batch, joint, m_channels});
    auto* dst = out->data<float>();
    std::fill_n(dst, out->get_size(), 0.0f);  // the text rows are not predictions

    const auto* latents = inputs.at("hidden_states")->data<float>();
    const auto* context = inputs.at("encoder_hidden_states")->data<float>();
    const auto* timestep = inputs.at("timestep")->data<float>();
    const size_t n_timesteps = inputs.at("timestep")->get_size();
    OPENVINO_ASSERT(n_timesteps == 1 || n_timesteps == batch, "[GGML] DiT: timestep must hold one value per sample");
    std::vector<float> planar(image * m_channels), result(model.output_size());
    for (size_t b = 0; b < batch; b++) {
        // token-major [image, C] -> stable-diffusion.cpp's planar [C, image]
        const float* lat = latents + b * image * m_channels;
        for (size_t t = 0; t < image; t++) {
            for (size_t c = 0; c < m_channels; c++) {
                planar[c * image + t] = lat[t * m_channels + c];
            }
        }
        model.write_input("dit_latent", planar.data(), planar.size() * sizeof(float));
        model.write_input("dit_context", context + b * text * m_context, text * m_context * sizeof(float));
        // GenAI hands over the flow timestep on [0, 1]; stable-diffusion.cpp's graph takes it on [0, 1000]
        const float ts = timestep[n_timesteps == 1 ? 0 : b] * 1000.0f;
        model.write_input("dit_timestep", &ts, sizeof(ts));
        OPENVINO_ASSERT(model.compute(), "[GGML] DiT compute failed");
        model.read_output(result.data());
        float* rows = dst + (b * joint + text) * m_channels;
        for (size_t t = 0; t < image; t++) {
            for (size_t c = 0; c < m_channels; c++) {
                rows[t * m_channels + c] = result[c * image + t];
            }
        }
    }
}

}  // namespace ggml_plugin
}  // namespace ov
