// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "encoder_runner.hpp"

#include <algorithm>
#include <cmath>

#include "openvino/core/except.hpp"

namespace ov {
namespace ggml_plugin {

namespace {

/// llama.cpp's llama_relative_position_bucket(x, y, n_buckets, bidirectional = true): T5's bucketed
/// relative position, exact up to n_buckets / 4 and logarithmic out to 128.
int32_t relative_bucket(int32_t x, int32_t y, int32_t n_buckets) {
    const int64_t max_distance = 128;
    n_buckets >>= 1;
    const int64_t max_exact = n_buckets >> 1;
    int32_t relative_position = x - y;
    int32_t bucket = (relative_position > 0) * n_buckets;
    relative_position = std::abs(relative_position);
    int32_t large = static_cast<int32_t>(std::floor(
        max_exact + std::log(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / std::log(1.0 * max_distance / max_exact)));
    large = std::min<int32_t>(large, n_buckets - 1);
    return bucket + (relative_position < max_exact ? relative_position : large);
}

}  // namespace

EncoderRunner::EncoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config) {
    m_graphs.push_back({model->input_size("inp_tokens"), model});
    for (const auto& path : config.buckets) {
        auto extra = GgmlModel::from_cgraph(path.string(), model);
        m_graphs.push_back({extra->input_size("inp_tokens"), extra});
    }
    std::sort(m_graphs.begin(), m_graphs.end(), [](const Graph& a, const Graph& b) {
        return a.tokens < b.tokens;
    });
    std::string arch;
    model->gguf_meta_str("general.architecture", arch);
    model->gguf_meta_i32(arch + ".attention.sliding_window", m_window);
    if (model->input_size("pos_bucket_enc") > 0) {
        model->gguf_meta_i32(arch + ".attention.relative_buckets_count", m_buckets);
        OPENVINO_ASSERT(m_buckets > 0, "[GGML] the encoder graph needs relative position buckets, but the GGUF gives no count");
    }
    const auto& g = m_graphs.front();
    // Per-token hidden states fill [tokens, hidden]; a classifier head leaves fewer values than tokens.
    m_per_token = g.model->output_size() >= g.tokens && g.model->output_size() % g.tokens == 0;
    m_out_per_seq = m_per_token ? g.model->output_size() / g.tokens : g.model->output_size();
    for (const auto& other : m_graphs) {
        OPENVINO_ASSERT(m_per_token ? other.model->output_size() == other.tokens * m_out_per_seq
                                    : other.model->output_size() == m_out_per_seq,
                        "[GGML] encoder graphs disagree on the output size");
    }
}

std::vector<PortDesc> EncoderRunner::inputs() const {
    return {{"input_ids", ov::element::i64, ov::PartialShape{-1, -1}},
            {"attention_mask", ov::element::i64, ov::PartialShape{-1, -1}},
            {"token_type_ids", ov::element::i64, ov::PartialShape{-1, -1}}};
}

std::vector<PortDesc> EncoderRunner::outputs() const {
    const auto n = static_cast<int64_t>(m_out_per_seq);
    if (m_per_token) {
        return {{"last_hidden_state", ov::element::f32, ov::PartialShape{-1, -1, n}}};
    }
    return {{"logits", ov::element::f32, ov::PartialShape{-1, n}}};
}

std::string EncoderRunner::backend_name() const {
    return m_graphs.front().model->backend_name();
}

void EncoderRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const auto shape = inputs.at("input_ids")->get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "[GGML] encoder input_ids must be [batch, seq]");
    const size_t batch = shape[0], seq = shape[1];
    const auto* ids = inputs.at("input_ids")->data<int64_t>();
    const auto* attn = inputs.at("attention_mask")->data<int64_t>();

    const auto it = std::find_if(m_graphs.begin(), m_graphs.end(), [&](const Graph& g) {
        return g.tokens >= seq;
    });
    OPENVINO_ASSERT(it != m_graphs.end(),
                    "[GGML] sequence of ",
                    seq,
                    " tokens exceeds the encoder graphs (largest: ",
                    m_graphs.back().tokens,
                    "); re-dump with a larger N_TOKENS or add GGML_BUCKETS");
    auto& model = *it->model;
    const size_t n = it->tokens;

    auto& out = outputs.at(m_per_token ? "last_hidden_state" : "logits");
    if (m_per_token) {
        out->set_shape({batch, seq, m_out_per_seq});
    } else {
        out->set_shape({batch, m_out_per_seq});
    }
    auto* dst = out->data<float>();

    const bool alibi = model.uses_alibi();
    const bool half_mask = model.input_nbytes("self_kq_mask") == n * n * sizeof(uint16_t);
    std::vector<float> mask_f(n * n);
    std::vector<uint16_t> mask_h(half_mask ? n * n : 0);
    std::vector<int32_t> tok(n), pos(n), out_ids(model.input_size("out_ids"));
    std::vector<float> hidden(model.output_size());
    for (size_t i = 0; i < out_ids.size(); i++) {
        out_ids[i] = static_cast<int32_t>(i);
    }
    // Layers with local attention take a second n x n mask (an unnamed leaf in the artifact): the same
    // padding mask, plus llama.cpp's symmetric window, |q - k| <= window / 2.
    std::vector<std::string> window_masks;
    for (const auto& name : model.input_names()) {
        if (name != "self_kq_mask" && name != "pos_bucket_enc" && model.input_size(name) == n * n) {
            window_masks.push_back(name);
        }
    }
    std::vector<float> window_f(window_masks.empty() ? 0 : n * n);
    std::vector<uint16_t> window_h(window_masks.empty() || !half_mask ? 0 : n * n);
    for (size_t b = 0; b < batch; b++) {
        const int64_t* row_ids = ids + b * seq;
        const int64_t* row_attn = attn + b * seq;
        // Positions count real tokens only, so a left-padded row starts at 0 like an unpadded one.
        int32_t next_pos = 0;
        for (size_t i = 0; i < n; i++) {
            const bool real = i < seq && row_attn[i] != 0;
            tok[i] = i < seq ? static_cast<int32_t>(row_ids[i]) : 0;
            pos[i] = real ? next_pos++ : 0;
        }
        // Row q (a query) sees key k when both are real tokens; any other row sees only itself, so a
        // padded row never softmaxes over nothing.
        for (size_t q = 0; q < n; q++) {
            const bool q_real = q < seq && row_attn[q] != 0;
            for (size_t k = 0; k < n; k++) {
                const bool k_real = k < seq && row_attn[k] != 0;
                // ALiBi models (Jina-BERT v2) carry the distance penalty in the mask: -|pos_q - pos_k|
                const float bias = alibi ? -static_cast<float>(std::abs(pos[q] - pos[k])) : 0.0f;
                mask_f[q * n + k] = (q_real ? k_real : k == q) ? bias : -INFINITY;
            }
        }
        model.write_input("inp_tokens", tok.data(), tok.size() * sizeof(int32_t));
        if (m_buckets > 0) {
            std::vector<int32_t> bucket(n * n);
            for (size_t j = 0; j < n; j++) {
                for (size_t i = 0; i < n; i++) {
                    bucket[j * n + i] = relative_bucket(pos[i], pos[j], m_buckets);
                }
            }
            model.write_input("pos_bucket_enc", bucket.data(), bucket.size() * sizeof(int32_t));
        }
        if (model.input_size("inp_pos") > 0) {
            model.write_input("inp_pos", pos.data(), pos.size() * sizeof(int32_t));
        }
        model.write_input("out_ids", out_ids.data(), out_ids.size() * sizeof(int32_t));
        if (half_mask) {
            std::transform(mask_f.begin(), mask_f.end(), mask_h.begin(), [](float v) {
                return to_half(v);
            });
            model.write_input("self_kq_mask", mask_h.data(), mask_h.size() * sizeof(uint16_t));
        } else {
            model.write_input("self_kq_mask", mask_f.data(), mask_f.size() * sizeof(float));
        }
        for (const auto& name : window_masks) {
            for (size_t q = 0; q < n; q++) {
                for (size_t k = 0; k < n; k++) {
                    const auto dist = std::abs(static_cast<int64_t>(q) - static_cast<int64_t>(k));
                    window_f[q * n + k] = (m_window > 0 && dist > m_window / 2) ? -INFINITY : mask_f[q * n + k];
                }
            }
            if (half_mask) {
                std::transform(window_f.begin(), window_f.end(), window_h.begin(), [](float v) {
                    return to_half(v);
                });
                model.write_input(name, window_h.data(), window_h.size() * sizeof(uint16_t));
            } else {
                model.write_input(name, window_f.data(), window_f.size() * sizeof(float));
            }
        }
        OPENVINO_ASSERT(model.compute(), "[GGML] encoder compute failed");
        model.read_output(hidden.data());
        if (m_per_token) {
            std::copy_n(hidden.data(), seq * m_out_per_seq, dst + b * seq * m_out_per_seq);
        } else {
            std::copy_n(hidden.data(), m_out_per_seq, dst + b * m_out_per_seq);
        }
    }
}

}  // namespace ggml_plugin
}  // namespace ov
