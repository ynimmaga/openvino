// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// EXPERIMENTAL. Executes a GGUF model on a ggml backend (Vulkan, CPU, CUDA, Metal, ...) using
// OpenVINO's native GGUF decoder builder as the graph description.
//
// The GGUF frontend builds a decoder topology once, from the .gguf tensor table, and drives a
// GraphEmitter to materialise it. The default emitter produces a GgufGraph for the OpenVINO op
// translators; the emitter here produces ggml tensors instead. Same builder, same architecture
// registry, same auto-detection -- a different execution target.
//
// LAYERING: openvino_gguf_frontend links no ggml. This component sits ON TOP of the frontend and
// is the only place ggml appears. Build it with -DENABLE_GGML_EMITTER=ON; it is OFF by default.
//
// STATUS: the frontend builds a single-token graph (T == 1), so prompt prefill runs one token per
// pass. MoE routing ops (MUL_MAT_ID / TOP_K / SOFT_MAX / SUM_ROWS / TRANSPOSE) are not mapped, so
// MoE architectures are rejected -- see unsupported_ops() after building.
#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "openvino/ggml_emitter/visibility.hpp"

struct ggml_tensor;
struct ggml_context;
struct ggml_cgraph;

namespace ov {
namespace ggml_emitter {

/// Every GGML op the emitter can translate. archprobe diffs an architecture's op inventory
/// against this set to report missing mappings by name rather than crashing on a null operand.
OV_GGML_EMITTER_API const std::set<std::string>& handled_ops();

/// A GGUF model built into a ggml graph, ready to execute on a ggml backend.
///
/// Owns the ggml contexts and the weight/graph buffers. `externals()` exposes the graph inputs
/// (tokens, positions, KV write index, attention mask) and the KV caches, which persist across
/// compute() calls so a generate loop can step the model.
class OV_GGML_EMITTER_API GgmlModel {
public:
    /// Build `gguf_path` for a context of `n_kv` slots. `rope_mode` of -1 derives the RoPE
    /// variant from the frontend's op_case (recommended); a non-negative value forces a ggml
    /// mode and exists for A/B testing the mapping.
    /// Throws if the architecture is unsupported or an op cannot be translated.
    static std::shared_ptr<GgmlModel> build(const std::string& gguf_path,
                                            int n_kv,
                                            const std::string& backend_name = "",
                                            int rope_mode = -1);
    ~GgmlModel();

    /// Run one forward pass. Inputs must already be written via write_input().
    bool compute();

    /// Copy `bytes` into the named graph input. Returns false if there is no such input --
    /// architectures differ in which of inp_tokens / inp_pos / inp_kv_idx / self_kq_mask they
    /// declare, so callers write opportunistically rather than assuming a fixed set.
    bool write_input(const std::string& name, const void* data, size_t bytes);

    /// Element count of a graph input, for sizing a mask or position buffer.
    size_t input_size(const std::string& name) const;

    /// Names of the graph inputs and KV caches.
    std::vector<std::string> input_names() const;

    /// Vocabulary size, i.e. the length of one logits row.
    size_t logits_size() const;

    /// Copy the logits row out after compute(). `out` must hold logits_size() floats.
    void read_logits(float* out) const;

    /// Graph inputs and KV caches as raw ggml handles, for callers that do link ggml.
    const std::map<std::string, ggml_tensor*>& externals() const;

    /// The logits tensor produced by the last graph node.
    ggml_tensor* logits() const;

    /// Ops the builder emitted that the emitter could not translate. Empty on success.
    const std::set<std::string>& unsupported_ops() const;

    size_t node_count() const;
    std::string backend_name() const;

private:
    GgmlModel();
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ggml_emitter
}  // namespace ov
