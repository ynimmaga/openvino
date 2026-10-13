// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_set.hpp"

namespace ov {
namespace ggml_plugin {

using ov::ggml_cgraph_loader::GgmlModel;

/// Causal LM decoder behind GenAI's paged attention interface (continuous batching): one call
/// carries the scheduled tokens of many sequences, each with its own block table.
///
/// The ggml graph's KV cache is the block pool: GenAI block b covers cache slots
/// [(b + 1) * 32, (b + 2) * 32); slots below 32 are scratch for padding rows. Every scheduled token
/// becomes a graph row that writes its K/V to its own slot and attends to its own sequence's slots
/// only, so rows of different sequences share one forward pass. Rows are packed into ALL_LOGITS
/// graphs of 1 / 4 / 8 rows, each pass on the smallest window holding the blocks it touches
/// (GGML_BUCKETS).
///
/// key_cache.0 / value_cache.0 hold GenAI's block bookkeeping, not K/V: each block carries its own
/// id, so a block GenAI copied (copy-on-write when sequences fork) shows up with another block's
/// id, and the runner then copies those slots in the ggml cache.
class PagedDecoderRunner : public Runner {
public:
    static constexpr size_t BLOCK_SIZE = 32;  // GenAI's KV block size for non-GPU devices

    PagedDecoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config);

    std::vector<PortDesc> inputs() const override;
    std::vector<PortDesc> outputs() const override;
    void infer(const TensorMap& inputs, const TensorMap& outputs) override;
    std::string backend_name() const override;
    /// KV_CACHE_MAX_BLOCKS: the block pool size, so GenAI's scheduler never hands out a block
    /// beyond the ggml cache.
    ov::AnyMap properties() const override;

private:
    struct Row {
        int64_t token = 0;
        int32_t position = 0;
        size_t slot = 0;
        size_t seq = 0;         // index into the step's sequences
        size_t seq_index = 0;   // position of the token within its sequence (past + i)
        int64_t out = -1;       // row of the logits output, or -1 if not sampled
    };

    void sync_forked_blocks(const ov::SoPtr<ov::ITensor>& key_cache, const std::vector<int32_t>& blocks);
    void forward(const std::vector<Row>& rows,
                 const GraphSet::Graph& graph,
                 const std::vector<std::vector<int32_t>>& tables,
                 float* logits);
    static size_t slot_of(const std::vector<int32_t>& table, size_t index) {
        return (table[index / BLOCK_SIZE] + 1) * BLOCK_SIZE + index % BLOCK_SIZE;
    }

    GraphSet m_graphs;
    size_t m_vocab = 0, m_pool_blocks = 0;
    std::vector<float> m_row_logits;
};

}  // namespace ggml_plugin
}  // namespace ov
