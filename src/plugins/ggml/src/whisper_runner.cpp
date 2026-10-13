// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "whisper_runner.hpp"

#include <algorithm>
#include <limits>

#include "openvino/core/except.hpp"
#include "sequence_state.hpp"

namespace ov {
namespace ggml_plugin {

namespace {

size_t meta(const GgmlModel& model, const char* key) {
    int32_t v = 0;
    OPENVINO_ASSERT(model.gguf_meta_i32(key, v) && v > 0, "[GGML] whisper .gguf is missing ", key);
    return static_cast<size_t>(v);
}

}  // namespace

// ------------------------------------------------------------------------------------------- encoder

WhisperEncoderRunner::WhisperEncoderRunner(std::shared_ptr<GgmlModel> model) : m_model(std::move(model)) {
    m_n_mels = meta(*m_model, "whisper.n_mels");
    m_n_ctx = meta(*m_model, "whisper.n_audio_ctx");
    m_n_state = meta(*m_model, "whisper.n_audio_state");
    m_frames = m_model->input_size("mel") / m_n_mels;
    OPENVINO_ASSERT(m_model->output_size() == m_n_ctx * m_n_state,
                    "[GGML] whisper encoder artifact must end at the encoder output; "
                    "dump it with SPLIT_CROSS=1");
}

std::vector<PortDesc> WhisperEncoderRunner::inputs() const {
    return {{"input_features",
             ov::element::f32,
             ov::PartialShape{-1, static_cast<int64_t>(m_n_mels), static_cast<int64_t>(m_frames)}}};
}

std::vector<PortDesc> WhisperEncoderRunner::outputs() const {
    return {{"last_hidden_state",
             ov::element::f32,
             ov::PartialShape{-1, static_cast<int64_t>(m_n_ctx), static_cast<int64_t>(m_n_state)}}};
}

void WhisperEncoderRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const auto& features = inputs.at("input_features");
    OPENVINO_ASSERT(features->get_shape() == ov::Shape({1, m_n_mels, m_frames}),
                    "[GGML] whisper encoder expects input_features [1, ",
                    m_n_mels,
                    ", ",
                    m_frames,
                    "], got ",
                    features->get_shape());
    OPENVINO_ASSERT(m_model->write_input("mel", features->data(), features->get_byte_size()),
                    "[GGML] whisper encoder artifact has no mel input");
    OPENVINO_ASSERT(m_model->compute(), "[GGML] whisper encoder compute failed");
    auto& hidden = outputs.at("last_hidden_state");
    hidden->set_shape({1, m_n_ctx, m_n_state});
    m_model->read_output(static_cast<float*>(hidden->data()));
}

std::string WhisperEncoderRunner::backend_name() const {
    return m_model->backend_name();
}

// ------------------------------------------------------------------------------------------- decoder

WhisperDecoderRunner::WhisperDecoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config)
    : m_model(std::move(model)) {
    OPENVINO_ASSERT(!config.cross.empty(),
                    "[GGML] the whisper decoder needs GGML_CROSS (whisper.cross.json from dump_whisper SPLIT_CROSS=1)");
    m_cross = GgmlModel::from_cgraph(config.cross.string(), config.weights.string(), config.backend);
    m_n_kv = m_model->input_size("KQ_mask");
    m_n_text_ctx = meta(*m_model, "whisper.n_text_ctx");
    m_n_audio_ctx = meta(*m_model, "whisper.n_audio_ctx");
    m_n_state = meta(*m_model, "whisper.n_audio_state");
    m_vocab = m_model->logits_size();
    OPENVINO_ASSERT(m_cross->input_size("embd_enc") == m_n_audio_ctx * m_n_state,
                    "[GGML] whisper cross artifact does not take the encoder output as embd_enc");
    m_state = {std::make_shared<SequenceState>("ggml_kv_cache", m_cached), nullptr};

    const std::string prefix = "cross_attention_qk_scaled_scores_";
    for (size_t l = 0;; l++) {
        const std::string name = prefix + std::to_string(l);
        if (m_model->named_output_size(name) == 0) {
            break;
        }
        m_scores.push_back(name);
    }
    if (!m_scores.empty()) {
        m_n_head = meta(*m_model, "whisper.n_text_head");
        m_frames_padded = m_model->named_output_size(m_scores[0]) / m_n_head;
        OPENVINO_ASSERT(m_frames_padded >= m_n_audio_ctx, "[GGML] cross-attention scores cover too few frames");
    }
}

std::vector<PortDesc> WhisperDecoderRunner::inputs() const {
    return {{"input_ids", ov::element::i64, ov::PartialShape{-1, -1}},
            {"encoder_hidden_states",
             ov::element::f32,
             ov::PartialShape{-1, static_cast<int64_t>(m_n_audio_ctx), static_cast<int64_t>(m_n_state)}},
            {"beam_idx", ov::element::i32, ov::PartialShape{-1}},
            {"cache_position", ov::element::i64, ov::PartialShape{-1}}};
}

std::vector<PortDesc> WhisperDecoderRunner::outputs() const {
    std::vector<PortDesc> out{{"logits", ov::element::f32, ov::PartialShape{-1, -1, static_cast<int64_t>(m_vocab)}}};
    for (const auto& name : m_scores) {
        out.push_back({name,
                       ov::element::f32,
                       ov::PartialShape{-1, static_cast<int64_t>(m_n_head), -1, static_cast<int64_t>(m_n_audio_ctx)}});
    }
    return out;
}

std::vector<ov::SoPtr<ov::IVariableState>> WhisperDecoderRunner::states() const {
    return {m_state};
}

std::string WhisperDecoderRunner::backend_name() const {
    return m_model->backend_name();
}

void WhisperDecoderRunner::project_cross(const float* encoder_hidden_states, size_t size) {
    OPENVINO_ASSERT(size == m_n_audio_ctx * m_n_state, "[GGML] encoder_hidden_states has ", size, " elements");
    m_cross->write_input("embd_enc", encoder_hidden_states, size * sizeof(float));
    OPENVINO_ASSERT(m_cross->compute(), "[GGML] whisper cross K/V compute failed");
    // The cross and decoder graphs own separate copies of the cross caches: a host round trip of a
    // few MB, once per audio window.
    for (const char* name : {"cache_k_cross", "cache_v_cross"}) {
        const size_t n = m_cross->input_nbytes(name);
        OPENVINO_ASSERT(n > 0 && n == m_model->input_nbytes(name), "[GGML] cross and decoder disagree on ", name);
        m_cross_buf.resize(n);
        m_cross->read_input(name, m_cross_buf.data(), n);
        m_model->write_input(name, m_cross_buf.data(), n);
    }
}

void WhisperDecoderRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const auto& ids = inputs.at("input_ids");
    OPENVINO_ASSERT(ids->get_shape().at(0) == 1, "[GGML] only batch size 1 is supported");
    const size_t n_new = ids->get_shape().at(1);

    size_t past = m_cached;
    const auto& cache_position = inputs.at("cache_position");
    if (cache_position->get_size() > 0) {
        past = static_cast<size_t>(static_cast<const int64_t*>(cache_position->data())[0]);
    }
    OPENVINO_ASSERT(past <= m_cached, "[GGML] cache_position ", past, " is past the ", m_cached, " cached tokens");
    OPENVINO_ASSERT(past + n_new <= m_n_text_ctx,
                    "[GGML] sequence length ",
                    past + n_new,
                    " exceeds whisper's text context (",
                    m_n_text_ctx,
                    ")");
    if (past == 0) {
        const auto& hidden = inputs.at("encoder_hidden_states");
        project_cross(static_cast<const float*>(hidden->data()), hidden->get_size());
    }

    const auto* tokens = static_cast<const int64_t*>(ids->data());
    for (const auto& name : m_scores) {
        outputs.at(name)->set_shape({1, m_n_head, n_new, m_n_audio_ctx});
    }
    for (size_t i = 0; i < n_new; i++) {
        const int32_t token = static_cast<int32_t>(tokens[i]);
        const int32_t pos = static_cast<int32_t>(past + i);
        const int64_t slot = pos;
        m_mask.assign(m_n_kv, -std::numeric_limits<float>::infinity());
        std::fill(m_mask.begin(), m_mask.begin() + pos + 1, 0.0f);
        m_model->write_input("embd", &token, sizeof(token));
        m_model->write_input("position", &pos, sizeof(pos));
        m_model->write_input("inp_kv_idx", &slot, sizeof(slot));
        m_model->write_input("KQ_mask", m_mask.data(), m_mask.size() * sizeof(float));
        OPENVINO_ASSERT(m_model->compute(), "[GGML] whisper decoder compute failed at position ", pos);
        for (const auto& name : m_scores) {
            m_score_buf.resize(m_n_head * m_frames_padded);
            m_model->read_named_output(name, m_score_buf.data());
            auto* dst = static_cast<float*>(outputs.at(name)->data());
            for (size_t h = 0; h < m_n_head; h++) {
                std::copy_n(m_score_buf.data() + h * m_frames_padded,
                            m_n_audio_ctx,
                            dst + (h * n_new + i) * m_n_audio_ctx);
            }
        }
    }
    m_cached = past + n_new;

    // Only the last row's logits are computed, as after GenAI's slice-before-matmul.
    auto& logits = outputs.at("logits");
    logits->set_shape({1, 1, m_vocab});
    m_model->read_logits(static_cast<float*>(logits->data()));
}

}  // namespace ggml_plugin
}  // namespace ov
