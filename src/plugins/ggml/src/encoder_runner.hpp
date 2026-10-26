// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_set.hpp"

namespace ov {
namespace ggml_plugin {

/// A bidirectional encoder (BERT / NomicBERT / Jina / ModernBERT / NeoBERT / T5 encoder / ...) dumped
/// with DUMP_EMBEDDINGS=none ALL_LOGITS=1: input_ids/attention_mask/token_type_ids -> last_hidden_state
/// [batch, seq, hidden], the interface OpenVINO GenAI's TextEmbeddingPipeline reads before it pools.
/// Artifacts for a classifier head (rerankers) emit one small vector per sequence instead, exposed as
/// "logits" [batch, n]. Several token counts may be given as GGML_BUCKETS; each sequence runs in the
/// smallest graph that holds it, padded and masked.
class EncoderRunner : public Runner {
public:
    EncoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::string backend_name() const override;

private:
    struct Graph {
        size_t tokens;
        std::shared_ptr<GgmlModel> model;
    };

    std::vector<Graph> m_graphs;  // ascending token count
    int32_t m_buckets = 0;        // T5 relative-attention buckets (0: the graph has no pos_bucket_enc input)
    int32_t m_window = 0;         // sliding-window attention span (0: none), for the extra n x n mask inputs
    size_t m_out_per_seq = 0;     // output values of the main graph per token (hidden) or per sequence (head)
    bool m_per_token = false;
};

}  // namespace ggml_plugin
}  // namespace ov
