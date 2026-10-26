// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runner.hpp"

#include <algorithm>
#include <sstream>

#include "decoder_runner.hpp"
#include "dit_runner.hpp"
#include "encoder_runner.hpp"
#include "paged_runner.hpp"
#include "vlm_runner.hpp"
#include "whisper_runner.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace ggml_plugin {

RunnerConfig RunnerConfig::parse(const ov::AnyMap& properties) {
    RunnerConfig config;
    if (auto it = properties.find("GGML_WEIGHTS"); it != properties.end()) {
        config.weights = it->second.as<std::string>();
    }
    if (auto it = properties.find("GGML_BACKEND"); it != properties.end()) {
        config.backend = it->second.as<std::string>();
    }
    if (auto it = properties.find("GGML_PAGED"); it != properties.end()) {
        const auto v = it->second.as<std::string>();
        config.paged = v == "YES" || v == "TRUE" || v == "true" || v == "1";
    }
    if (auto it = properties.find("GGML_CROSS"); it != properties.end()) {
        config.cross = it->second.as<std::string>();
    }
    // "4:/path/decoder_b4.json;8:/path/decoder_b8.json"
    if (auto it = properties.find("GGML_BUCKETS"); it != properties.end()) {
        std::stringstream ss(it->second.as<std::string>());
        std::string entry;
        while (std::getline(ss, entry, ';')) {
            if (entry.empty()) {
                continue;
            }
            // "<tokens>:<artifact>" is accepted too; the token count and window come from the artifact.
            const auto colon = entry.find(':');
            const bool counted = colon != std::string::npos && colon > 0 &&
                                 entry.find_first_not_of("0123456789") == colon;
            config.buckets.emplace_back(counted ? entry.substr(colon + 1) : entry);
        }
    }
    return config;
}

std::shared_ptr<Runner> make_runner(const std::filesystem::path& artifact, const RunnerConfig& config) {
    OPENVINO_ASSERT(!config.weights.empty(), "[GGML] GGML_WEIGHTS (the .gguf with the model weights) is required");
    auto model = GgmlModel::from_cgraph(artifact.string(), config.weights.string(), config.backend);
    const auto names = model->input_names();
    const auto has = [&](const char* name) {
        return std::find(names.begin(), names.end(), name) != names.end();
    };
    // stable-diffusion.cpp's Qwen-Image 2.1 transformer, with its inputs named by the capture hook
    if (has("dit_latent") && has("dit_context") && has("dit_pe")) {
        return std::make_shared<DitRunner>(std::move(model), config);
    }
    // A causal decoder writes a KV cache (inp_kv_idx); positions may be absent (ALiBi).
    // Recurrent (Mamba / RWKV) layers keep per-sequence state selected by rs_s_copy instead.
    const bool decoder = (has("inp_kv_idx") || has("rs_s_copy")) && !has("KQ_mask");
    if (decoder && config.paged) {
        return std::make_shared<PagedDecoderRunner>(std::move(model), config);
    }
    if (decoder) {
        return std::make_shared<DecoderRunner>(std::move(model), config);
    }
    if (has("mel")) {
        return std::make_shared<WhisperEncoderRunner>(std::move(model));
    }
    if (has("KQ_mask") && has("position")) {
        return std::make_shared<WhisperDecoderRunner>(std::move(model), config);
    }
    if (has("inp_raw")) {
        return std::make_shared<VisionRunner>(std::move(model));
    }
    // A bidirectional encoder: token ids, positions and a full (non-causal) mask, no KV cache.
    if (has("inp_tokens") && has("self_kq_mask") && has("out_ids")) {
        return std::make_shared<EncoderRunner>(std::move(model), config);
    }
    if (has("inp_tokens") && model->node_count() <= 2) {
        return std::make_shared<TokenEmbeddingsRunner>(std::move(model));
    }
    OPENVINO_THROW("[GGML] unrecognized artifact ", artifact, ": no runner handles its inputs");
}

}  // namespace ggml_plugin
}  // namespace ov
