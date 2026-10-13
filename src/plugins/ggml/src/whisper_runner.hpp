// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"
#include "runner.hpp"

namespace ov {
namespace ggml_plugin {

using ov::ggml_cgraph_loader::GgmlModel;

/// whisper.encoder.json (dump_whisper SPLIT_CROSS=1) as an OpenVINO Whisper encoder:
/// input_features [1, n_mels, 2 * n_audio_ctx] -> last_hidden_state [1, n_audio_ctx, n_audio_state].
class WhisperEncoderRunner : public Runner {
public:
    explicit WhisperEncoderRunner(std::shared_ptr<GgmlModel> model);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::string backend_name() const override;

private:
    std::shared_ptr<GgmlModel> m_model;
    size_t m_n_mels = 0, m_frames = 0, m_n_ctx = 0, m_n_state = 0;
};

/// whisper.decoder.json + whisper.cross.json as a stateful OpenVINO Whisper decoder:
/// input_ids, encoder_hidden_states, beam_idx, cache_position -> logits. The cross-attention K/V
/// are projected from encoder_hidden_states (cross graph) at the start of every sequence.
class WhisperDecoderRunner : public Runner {
public:
    WhisperDecoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::vector<ov::SoPtr<ov::IVariableState>> states() const override;
    std::string backend_name() const override;

private:
    void project_cross(const float* encoder_hidden_states, size_t size);

    std::shared_ptr<GgmlModel> m_model;
    std::shared_ptr<GgmlModel> m_cross;
    std::vector<uint8_t> m_cross_buf;
    std::vector<float> m_mask;
    size_t m_n_kv = 0, m_n_text_ctx = 0, m_n_audio_ctx = 0, m_n_state = 0, m_vocab = 0;
    // Cross-attention score outputs (dump_whisper CROSS_QK=1), one per decoder layer, for word
    // timestamps: graph rows are [n_head][frames_padded]; the port drops the padding frames.
    std::vector<std::string> m_scores;
    size_t m_n_head = 0, m_frames_padded = 0;
    std::vector<float> m_score_buf;
    size_t m_cached = 0;
    ov::SoPtr<ov::IVariableState> m_state;
};

}  // namespace ggml_plugin
}  // namespace ov
