// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "openvino/runtime/tensor.hpp"
#include "quant/gguf.hpp"

#include "openvino/frontend/gguf/visibility.hpp"
#include <string>

namespace ov {
namespace frontend {
namespace gguf {

struct GgufGraph;
class GraphEmitter;  // defined in gguf_graph.hpp; only used here as a shared_ptr return type

// Build a GgufGraph natively from a .gguf file (no llama.cpp / gguf dependency).
// Parses the container, then dispatches to a per-architecture builder that emits nodes in
// the GGML op vocabulary reproducing llama.cpp's cgraph topology for that architecture.
// Throws if the architecture is not supported natively.
std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file);

// Same, but driving a caller-supplied emitter (e.g. one that emits ggml tensors directly).
using EmitterFactory = std::function<std::unique_ptr<GraphEmitter>(
    std::unordered_map<std::string, ov::Tensor>&, std::unordered_map<std::string, GgufTensorType>&,
    const std::string& arch)>;

GGUF_FRONTEND_API std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file,
                                                                        const EmitterFactory& make_emitter);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
