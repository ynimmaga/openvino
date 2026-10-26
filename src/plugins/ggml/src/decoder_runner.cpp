// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Causal LM decoder (dump_cgraph, optionally DUMP_VLM_DECODER) behind the interface of a stateful
// OpenVINO LLM: input_ids (or inputs_embeds), attention_mask, position_ids, beam_idx -> logits,
// with the KV cache held as state between calls.

#include "decoder_runner.hpp"

#include <algorithm>
#include <cstring>

#include "openvino/core/except.hpp"
#include "sequence_state.hpp"

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

}  // namespace

DecoderRunner::DecoderRunner(std::shared_ptr<GgmlModel> model, const RunnerConfig& config)
    : m_graphs(std::move(model), config) {
    const auto& main = m_graphs.main();
    m_vocab = main.logits_size();
    const auto names = main.input_names();
    m_embeds = std::find(names.begin(), names.end(), "embd") != names.end();
    if (m_embeds) {
        m_hidden = main.input_size("embd") / m_graphs.graphs().front().tokens;
    }
    // Padding rows of a partially filled graph write their K/V to the bottom slots, which every
    // window contains; sequence index i lives in slot m_scratch_slots + i.
    const auto all_inputs = main.input_names();
    m_recurrent = main.has_recurrent_state();
    OPENVINO_ASSERT(!m_recurrent || m_graphs.max_tokens() == 1,
                    "[GGML] recurrent models take 1-token graphs only: padding rows would enter the recurrent state (graph has ",
                    m_graphs.max_tokens(),
                    " rows)");
    {
        std::string arch;
        main.gguf_meta_str("general.architecture", arch);
        int32_t ngram = 0, per_gram = 0, eos = 0;
        if (main.gguf_meta_i32(arch + ".ple.ngram_size", ngram) && ngram > 1) {
            main.gguf_meta_i32(arch + ".ple.heads_per_ngram", per_gram);
            main.gguf_meta_i32(arch + ".ple.eos_token_id", eos);
            OPENVINO_ASSERT(main.gguf_meta_u64_array(arch + ".ple.layer_multipliers", m_ple.multipliers) &&
                                main.gguf_meta_u64_array(arch + ".ple.head_offsets", m_ple.offsets) &&
                                main.gguf_meta_u64_array(arch + ".ple.head_vocab_sizes", m_ple.vocab),
                            "[GGML] the GGUF declares n-gram embeddings but not their hash tables");
            m_ple.ngram = ngram;
            m_ple.per_gram = per_gram;
            m_ple.eos = eos;
            m_ple.n_heads = (ngram - 1) * per_gram;
            // the index input is the one i32 graph input of n_heads * rows values that has no name of its own
            for (const auto& name : names) {
                if (starts_with(name, "leaf_") &&
                    main.input_size(name) == static_cast<size_t>(m_ple.n_heads) * m_graphs.graphs().front().tokens) {
                    m_ple.input = name;
                }
            }
            OPENVINO_ASSERT(!m_ple.input.empty(), "[GGML] cannot find the n-gram embedding index input in the artifact");
        }
    }
    m_scratch_slots = m_graphs.max_tokens() - 1;
    OPENVINO_ASSERT(m_scratch_slots < m_graphs.max_n_kv(), "[GGML] largest graph exceeds the context size");
    m_state = {std::make_shared<SequenceState>("ggml_kv_cache", m_cached), nullptr};
}

std::vector<PortDesc> DecoderRunner::inputs() const {
    std::vector<PortDesc> in;
    if (m_embeds) {
        in.push_back({"inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, static_cast<int64_t>(m_hidden)}});
    } else {
        in.push_back({"input_ids", ov::element::i64, ov::PartialShape{-1, -1}});
    }
    in.push_back({"attention_mask", ov::element::i64, ov::PartialShape{-1, -1}});
    in.push_back({"position_ids", ov::element::i64, ov::PartialShape{-1, -1}});
    in.push_back({"beam_idx", ov::element::i32, ov::PartialShape{-1}});
    return in;
}

std::vector<PortDesc> DecoderRunner::outputs() const {
    return {{"logits", ov::element::f32, ov::PartialShape{-1, -1, static_cast<int64_t>(m_vocab)}}};
}

std::vector<ov::SoPtr<ov::IVariableState>> DecoderRunner::states() const {
    return {m_state};
}

std::string DecoderRunner::backend_name() const {
    return m_graphs.main().backend_name();
}

void DecoderRunner::forward(const Rows& rows, const GraphSet::Graph& graph) {
    auto& model = graph.model;
    m_active = model;
    const size_t bucket = graph.tokens;
    const size_t chunk = rows.positions.size();
    const bool alibi = model->uses_alibi();

    // M-RoPE: the sections are stored one after another; a text token repeats its position in the first
    // three and has 0 in a fourth.
    const size_t n_pos = GraphSet::positions_per_token(*model);
    std::vector<int32_t> pos(bucket * n_pos, 0);
    std::vector<int64_t> slot(bucket, 0);
    for (size_t i = 0; i < chunk; i++) {
        for (size_t j = 0; j < n_pos; j++) {
            // llama.cpp: a text token's four M-RoPE streams are [pos, pos, pos, 0]
            pos[j * bucket + i] = (n_pos == 4 && j == 3) ? 0 : rows.positions[i];
        }
        slot[i] = static_cast<int64_t>(m_scratch_slots + rows.first_index + i);
    }
    for (size_t i = chunk; i < bucket; i++) {
        slot[i] = static_cast<int64_t>(i - chunk);
    }
    if (m_embeds) {
        std::vector<float> embd(bucket * m_hidden, 0.0f);
        std::memcpy(embd.data(), rows.embeds, chunk * m_hidden * sizeof(float));
        model->write_input("embd", embd.data(), embd.size() * sizeof(float));
    } else {
        std::vector<int32_t> tok(bucket, 0);
        for (size_t i = 0; i < chunk; i++) {
            tok[i] = static_cast<int32_t>(rows.tokens[i]);
        }
        for (const auto& name : model->input_names()) {
            if (starts_with(name, "inp_tokens")) {  // every token-id input carries the same ids
                model->write_input(name, tok.data(), tok.size() * sizeof(int32_t));
            }
        }
    }
    // llama.cpp's mixed token/embedding path is built into every graph but selected away for token batches;
    // its inputs still need in-range values (slot 0, token 0) so SET_ROWS and GET_ROWS stay valid.
    for (const auto& name : model->input_names()) {
        if (starts_with(name, "inp_mixed")) {
            const std::vector<uint8_t> zeros(model->input_nbytes(name), 0);
            model->write_input(name, zeros.data(), zeros.size());
        }
    }
    if (!m_ple.input.empty()) {
        // llama.cpp's llm_graph_input_qwen4exp_ple: hash the token with its predecessors (an EOS, or the
        // start of the sequence, cuts the context) for each n-gram size and head
        const int64_t n_prev = m_ple.ngram - 1;
        std::vector<int32_t> idx(static_cast<size_t>(m_ple.n_heads) * bucket, 0);
        for (size_t i = 0; i < chunk; i++) {
            const int64_t at = static_cast<int64_t>(rows.first_index + i);
            std::vector<int64_t> ctx(m_ple.ngram);
            ctx[0] = m_history[at];
            bool cut = false;
            for (int64_t s = 1; s <= n_prev; s++) {
                const int64_t t = (cut || at - s < 0) ? -1 : m_history[at - s];
                cut = cut || t < 0 || t == m_ple.eos;
                ctx[s] = cut ? m_ple.eos : t;
            }
            for (int64_t n = 2; n <= m_ple.ngram; n++) {
                uint64_t mixed = static_cast<uint64_t>(ctx[0]) * m_ple.multipliers[0];
                for (int64_t j = 1; j < n; j++) {
                    mixed ^= static_cast<uint64_t>(ctx[j]) * m_ple.multipliers[j];
                }
                const int64_t base = (n - 2) * m_ple.per_gram;
                for (int64_t g = 0; g < m_ple.per_gram; g++) {
                    const size_t h = static_cast<size_t>(base + g);
                    idx[i * m_ple.n_heads + h] = static_cast<int32_t>(mixed % m_ple.vocab[h] + m_ple.offsets[h]);
                }
            }
        }
        model->write_input(m_ple.input, idx.data(), idx.size() * sizeof(int32_t));
    }
    if (n_pos > 0) {
        model->write_input("inp_pos", pos.data(), pos.size() * sizeof(int32_t));
    }
    m_graphs.write_attn_scale(*model, rows.positions);
    const int32_t out_id = static_cast<int32_t>(chunk) - 1;
    model->write_input("inp_out_ids", &out_id, sizeof(out_id));

    for (const auto& name : model->input_names()) {
        if (starts_with(name, "inp_kv_idx")) {
            model->write_input(name, slot.data(), slot.size() * sizeof(int64_t));
        } else if (is_mask_input(name)) {
            // Row r (sequence index first_index + r) sees every earlier token the attention mask
            // keeps; a padding row sees only its own scratch slot.
            std::vector<uint16_t> mask(model->input_size(name), FP16_NEG_INF);
            for (size_t r = 0; r < bucket; r++) {
                uint16_t* row = mask.data() + r * graph.n_kv;
                if (r >= chunk) {
                    row[slot[r]] = 0;
                    continue;
                }
                const size_t q = rows.first_index + r;
                for (size_t c = 0; c <= q; c++) {
                    if (rows.mask[c] != 0) {
                        row[m_scratch_slots + c] = alibi ? to_half(-static_cast<float>(q - c)) : 0;
                    }
                }
            }
            model->write_input(name, mask.data(), mask.size() * sizeof(uint16_t));
        }
    }
    OPENVINO_ASSERT(model->compute(), "[GGML] decoder compute failed at index ", rows.first_index);
}

void DecoderRunner::infer(const TensorMap& inputs, const TensorMap& outputs) {
    const auto& ids = inputs.at(m_embeds ? "inputs_embeds" : "input_ids");
    const auto& mask = inputs.at("attention_mask");
    const auto ids_shape = ids->get_shape();
    OPENVINO_ASSERT(ids_shape.at(0) == 1,
                    "[GGML] the stateful decoder holds one sequence (batch ",
                    ids_shape.at(0),
                    " requested). For batches, beam search or num_return_sequences > 1, compile with "
                    "GGML_PAGED=YES and use LLMPipeline(compiled_model, tokenizer, SchedulerConfig) or "
                    "ContinuousBatchingPipeline");
    const size_t n_new = ids_shape.at(1);
    const size_t total = mask->get_shape().at(1);
    OPENVINO_ASSERT(n_new > 0 && total >= n_new, "[GGML] attention_mask is shorter than the new tokens");
    const size_t past = total - n_new;
    OPENVINO_ASSERT(past <= m_cached,
                    "[GGML] attention_mask covers ",
                    past,
                    " past tokens but the KV cache holds ",
                    m_cached);
    if (m_recurrent) {
        OPENVINO_ASSERT(past == 0 || past == m_cached,
                        "[GGML] the recurrent state cannot be rewound (",
                        m_cached,
                        " tokens folded in, ",
                        past,
                        " kept); chat history reuse needs the full history re-sent");
        if (past == 0) {
            m_graphs.main().zero_recurrent_state();
        }
    }
    const size_t usable = m_graphs.max_n_kv() - m_scratch_slots;
    OPENVINO_ASSERT(total <= usable,
                    "[GGML] sequence length ",
                    total,
                    " exceeds the largest context the artifacts were dumped for (",
                    usable,
                    " usable of ",
                    m_graphs.max_n_kv(),
                    " slots); re-dump with a larger n_kv");

    if (auto it = inputs.find("beam_idx"); it != inputs.end() && it->second->get_size() > 0) {
        OPENVINO_ASSERT(it->second->get_size() == 1 && static_cast<const int32_t*>(it->second->data())[0] == 0,
                        "[GGML] the stateful decoder cannot reorder beams; for beam search compile with "
                        "GGML_PAGED=YES and use LLMPipeline(compiled_model, tokenizer, SchedulerConfig)");
    }
    std::vector<int32_t> positions(n_new);
    const auto pos_it = inputs.find("position_ids");
    const bool have_pos = pos_it != inputs.end() && pos_it->second->get_size() == n_new;
    for (size_t i = 0; i < n_new; i++) {
        positions[i] = have_pos ? static_cast<int32_t>(static_cast<const int64_t*>(pos_it->second->data())[i])
                                : static_cast<int32_t>(past + i);
    }

    if (!m_ple.input.empty()) {
        OPENVINO_ASSERT(!m_embeds, "[GGML] n-gram embeddings with embedding inputs are not supported");
        if (past == 0) {
            m_history.clear();
        }
        OPENVINO_ASSERT(m_history.size() >= past, "[GGML] the n-gram history is shorter than the kept context");
        m_history.resize(past);
        const auto* new_ids = static_cast<const int64_t*>(ids->data());
        m_history.insert(m_history.end(), new_ids, new_ids + n_new);
    }
    const auto* mask_data = static_cast<const int64_t*>(mask->data());
    size_t done = 0;
    while (done < n_new) {
        const size_t first = past + done;
        const auto& graph = m_graphs.pick(n_new - done, [&](size_t rows) {
            return m_scratch_slots + first + rows;
        });
        const size_t chunk = std::min(n_new - done, graph.tokens);
        Rows rows;
        rows.first_index = first;
        rows.positions.assign(positions.begin() + done, positions.begin() + done + chunk);
        rows.mask = mask_data;
        if (m_embeds) {
            rows.embeds = static_cast<const float*>(ids->data()) + done * m_hidden;
        } else {
            rows.tokens = static_cast<const int64_t*>(ids->data()) + done;
        }
        forward(rows, graph);
        done += chunk;
    }
    m_cached = total;

    // Only the last row's logits are computed, as after GenAI's slice-before-matmul.
    auto& logits = outputs.at("logits");
    logits->set_shape({1, 1, m_vocab});
    m_active->read_logits(static_cast<float*>(logits->data()));
}

}  // namespace ggml_plugin
}  // namespace ov
