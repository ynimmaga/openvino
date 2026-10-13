// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_runner.hpp"

#include <algorithm>
#include <cstring>

#include "openvino/core/except.hpp"

namespace ov {
namespace ggml_plugin {

namespace {

constexpr uint16_t FP16_NEG_INF = 0xFC00;

bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool is_mask_input(const std::string& name) {
    return starts_with(name, "self_kq_mask") || starts_with(name, "attn_inp_kq_mask");
}

template <typename T>
const T* data_of(const TensorMap& inputs, const char* name) {
    return static_cast<const T*>(inputs.at(name)->data());
}

}  // namespace

PagedDecoderRunner::PagedDecoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config)
    : m_graphs(std::move(model), config) {
    m_vocab = m_graphs.main().logits_size();
    for (const auto& g : m_graphs.graphs()) {
        OPENVINO_ASSERT(g.model->output_size() == g.tokens * m_vocab,
                        "[GGML] paged decoder graphs must be dumped with ALL_LOGITS=1 (",
                        g.tokens,
                        "-token graph is not): rows of different sequences each need their logits");
    }
    OPENVINO_ASSERT(m_graphs.max_tokens() <= BLOCK_SIZE, "[GGML] paged decoder graphs take at most ", BLOCK_SIZE, " rows");
    OPENVINO_ASSERT(m_graphs.max_n_kv() >= 2 * BLOCK_SIZE, "[GGML] paged decoder: n_kv is too small");
    m_pool_blocks = m_graphs.max_n_kv() / BLOCK_SIZE - 1;  // block 0 holds the scratch slots
}

std::vector<PortDesc> PagedDecoderRunner::inputs() const {
    const ov::PartialShape blocks{-1, 1, static_cast<int64_t>(BLOCK_SIZE), 1};
    return {{"input_ids", ov::element::i64, ov::PartialShape{-1}},
            {"position_ids", ov::element::i64, ov::PartialShape{-1}},
            {"past_lens", ov::element::i32, ov::PartialShape{-1}},
            {"subsequence_begins", ov::element::i32, ov::PartialShape{-1}},
            {"block_indices", ov::element::i32, ov::PartialShape{-1}},
            {"block_indices_begins", ov::element::i32, ov::PartialShape{-1}},
            {"max_context_len", ov::element::i32, ov::PartialShape{}},
            {"sampled_tokens_indices", ov::element::i64, ov::PartialShape{-1}},
            {"key_cache.0", ov::element::u8, blocks},
            {"value_cache.0", ov::element::u8, blocks}};
}

std::vector<PortDesc> PagedDecoderRunner::outputs() const {
    return {{"logits", ov::element::f32, ov::PartialShape{-1, 1, static_cast<int64_t>(m_vocab)}}};
}

ov::AnyMap PagedDecoderRunner::properties() const {
    return {{"KV_CACHE_MAX_BLOCKS", m_pool_blocks}};
}

std::string PagedDecoderRunner::backend_name() const {
    return m_graphs.main().backend_name();
}

void PagedDecoderRunner::sync_forked_blocks(const ov::SoPtr<ov::ITensor>& key_cache,
                                            const std::vector<int32_t>& blocks) {
    // Each bookkeeping block starts with (its own block id + 1); 0 marks a block GenAI has just
    // allocated. Any other value is the id of the block it was copied from.
    auto* bytes = static_cast<uint8_t*>(key_cache->data());
    const size_t n_alloc = key_cache->get_shape().at(0);
    for (const int32_t b : blocks) {
        OPENVINO_ASSERT(b >= 0 && static_cast<size_t>(b) < m_pool_blocks && static_cast<size_t>(b) < n_alloc,
                        "[GGML] KV block ",
                        b,
                        " is outside the ggml cache's ",
                        m_pool_blocks,
                        " blocks; set SchedulerConfig::num_kv_blocks <= ",
                        m_pool_blocks,
                        " (or re-dump with a larger n_kv)");
        uint32_t tag = 0;
        std::memcpy(&tag, bytes + b * BLOCK_SIZE, sizeof(tag));
        const uint32_t own = static_cast<uint32_t>(b) + 1;
        if (tag != 0 && tag != own) {
            m_graphs.main().copy_cache_slots(tag * BLOCK_SIZE, (b + 1) * BLOCK_SIZE, BLOCK_SIZE);
        }
        std::memcpy(bytes + b * BLOCK_SIZE, &own, sizeof(own));
    }
}

void PagedDecoderRunner::forward(const std::vector<Row>& rows,
                                 const GraphSet::Graph& graph,
                                 const std::vector<std::vector<int32_t>>& tables,
                                 float* logits) {
    auto& model = graph.model;
    const size_t bucket = graph.tokens;
    const size_t n = rows.size();
    const size_t n_pos = GraphSet::positions_per_token(*model);
    std::vector<int32_t> tok(bucket, 0), pos(bucket * n_pos, 0);
    std::vector<int64_t> slot(bucket, 0);
    for (size_t i = 0; i < bucket; i++) {
        // Padding rows write to (and only see) a scratch slot below the pool.
        slot[i] = static_cast<int64_t>(i < n ? rows[i].slot : i - n);
        if (i < n) {
            tok[i] = static_cast<int32_t>(rows[i].token);
            for (size_t j = 0; j < n_pos; j++) {
                pos[j * bucket + i] = rows[i].position;  // same position in every M-RoPE section
            }
        }
    }
    model->write_input("inp_tokens", tok.data(), tok.size() * sizeof(int32_t));
    if (n_pos > 0) {
        model->write_input("inp_pos", pos.data(), pos.size() * sizeof(int32_t));
    }
    std::vector<int32_t> out_ids(model->input_size("inp_out_ids"));
    for (size_t i = 0; i < out_ids.size(); i++) {
        out_ids[i] = static_cast<int32_t>(i);
    }
    model->write_input("inp_out_ids", out_ids.data(), out_ids.size() * sizeof(int32_t));
    for (const auto& name : model->input_names()) {
        if (starts_with(name, "inp_kv_idx")) {
            model->write_input(name, slot.data(), slot.size() * sizeof(int64_t));
        } else if (is_mask_input(name)) {
            std::vector<uint16_t> mask(model->input_size(name), FP16_NEG_INF);
            for (size_t r = 0; r < bucket; r++) {
                uint16_t* row = mask.data() + r * graph.n_kv;
                if (r >= n) {
                    row[slot[r]] = 0;
                    continue;
                }
                const auto& table = tables[rows[r].seq];
                for (size_t p = 0; p <= rows[r].seq_index; p++) {
                    row[slot_of(table, p)] = 0;
                }
            }
            model->write_input(name, mask.data(), mask.size() * sizeof(uint16_t));
        }
    }
    OPENVINO_ASSERT(model->compute(), "[GGML] paged decoder compute failed");

    const bool any_sampled = std::any_of(rows.begin(), rows.end(), [](const Row& r) {
        return r.out >= 0;
    });
    if (!any_sampled) {
        return;
    }
    m_row_logits.resize(model->output_size());
    model->read_output(m_row_logits.data());
    for (size_t r = 0; r < n; r++) {
        if (rows[r].out >= 0) {
            std::memcpy(logits + rows[r].out * m_vocab, m_row_logits.data() + r * m_vocab, m_vocab * sizeof(float));
        }
    }
}

void PagedDecoderRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const size_t n_tokens = inputs.at("input_ids")->get_size();
    const size_t n_seqs = inputs.at("past_lens")->get_size();
    const auto* ids = data_of<int64_t>(inputs, "input_ids");
    const auto* positions = data_of<int64_t>(inputs, "position_ids");
    const auto* past_lens = data_of<int32_t>(inputs, "past_lens");
    const auto* begins = data_of<int32_t>(inputs, "subsequence_begins");
    const auto* blocks = data_of<int32_t>(inputs, "block_indices");
    const auto* block_begins = data_of<int32_t>(inputs, "block_indices_begins");

    std::vector<std::vector<int32_t>> tables(n_seqs);
    for (size_t s = 0; s < n_seqs; s++) {
        tables[s].assign(blocks + block_begins[s], blocks + block_begins[s + 1]);
        const size_t needed = (past_lens[s] + (begins[s + 1] - begins[s]) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        OPENVINO_ASSERT(tables[s].size() >= needed,
                        "[GGML] sequence ",
                        s,
                        " has ",
                        tables[s].size(),
                        " KV blocks for ",
                        needed,
                        " needed; the KV block size must be ",
                        BLOCK_SIZE);
        sync_forked_blocks(inputs.at("key_cache.0"), tables[s]);
    }

    // Logits only for the rows GenAI samples from (all rows when it does not say).
    const auto& sampled = inputs.at("sampled_tokens_indices");
    std::vector<int64_t> out_row(n_tokens, -1);
    size_t n_out = 0;
    if (sampled->get_size() > 0) {
        const auto* idx = static_cast<const int64_t*>(sampled->data());
        for (size_t i = 0; i < sampled->get_size(); i++) {
            out_row.at(idx[i]) = static_cast<int64_t>(n_out++);
        }
    } else {
        for (size_t t = 0; t < n_tokens; t++) {
            out_row[t] = static_cast<int64_t>(n_out++);
        }
    }
    auto& logits = outputs.at("logits");
    logits->set_shape({n_out, 1, m_vocab});

    std::vector<Row> all(n_tokens);
    for (size_t s = 0; s < n_seqs; s++) {
        for (int32_t t = begins[s]; t < begins[s + 1]; t++) {
            Row& r = all[t];
            r.token = ids[t];
            r.position = static_cast<int32_t>(positions[t]);
            r.seq = s;
            r.seq_index = static_cast<size_t>(past_lens[s] + (t - begins[s]));
            r.slot = slot_of(tables[s], r.seq_index);
            r.out = out_row[t];
        }
    }
    // Pack rows into graphs; a later row of the same sequence sees an earlier one's K/V because
    // each graph writes its rows' K/V before attending.
    size_t done = 0;
    while (done < n_tokens) {
        // The window must hold every block the chunk's rows write or attend to.
        const auto& graph = m_graphs.pick(n_tokens - done, [&](size_t chunk) {
            size_t top = 0;
            for (size_t t = done; t < done + chunk; t++) {
                const auto& table = tables[all[t].seq];
                const size_t last_block = all[t].seq_index / BLOCK_SIZE;
                for (size_t k = 0; k <= last_block; k++) {
                    top = std::max(top, static_cast<size_t>(table[k]) + 2);
                }
            }
            return top * BLOCK_SIZE;
        });
        const size_t chunk = std::min(n_tokens - done, graph.tokens);
        std::vector<Row> rows(all.begin() + done, all.begin() + done + chunk);
        forward(rows, graph, tables, static_cast<float*>(logits->data()));
        done += chunk;
    }
}

}  // namespace ggml_plugin
}  // namespace ov
