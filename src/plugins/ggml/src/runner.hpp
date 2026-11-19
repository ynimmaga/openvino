// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace ggml_cgraph_loader {
class GgmlModel;
}  // namespace ggml_cgraph_loader

namespace ggml_plugin {

struct PortDesc {
    std::string name;
    ov::element::Type type;
    ov::PartialShape shape;
};

using TensorMap = std::map<std::string, ov::SoPtr<ov::ITensor>>;

/// Executes one cgraph artifact behind an OpenVINO-style interface. Each runner translates the
/// inputs/outputs an OpenVINO model of the same role would have (e.g. input_ids/attention_mask ->
/// logits for a stateful LLM) into the ggml graph inputs the artifact was dumped with.
class Runner {
public:
    virtual ~Runner() = default;

    virtual std::vector<PortDesc> inputs() const = 0;
    virtual std::vector<PortDesc> outputs() const = 0;

    /// `outputs` holds one tensor per output port; the runner reshapes and fills it.
    virtual void infer(const TensorMap& inputs, const TensorMap& outputs) = 0;

    virtual std::vector<ov::SoPtr<ov::IVariableState>> states() const {
        return {};
    }

    virtual std::string backend_name() const = 0;

    /// Read-only compiled model properties specific to this runner (e.g. KV cache capacity).
    virtual ov::AnyMap properties() const {
        return {};
    }
};

/// Properties accepted by core.compile_model(<artifact.json>, "GGML", ...).
struct RunnerConfig {
    std::filesystem::path weights;  // GGML_WEIGHTS: the .gguf the artifact reads its weights from
    std::string backend;            // GGML_BACKEND: ggml device name ("Vulkan0", "CPU"); empty = auto
    std::vector<std::filesystem::path> buckets;  // GGML_BUCKETS: more graphs of the model (token counts, windows)
    std::filesystem::path cross;                      // GGML_CROSS: whisper cross-attention K/V graph
    bool paged = false;  // GGML_PAGED: decoder with GenAI's paged attention (continuous batching) interface

    static RunnerConfig parse(const ov::AnyMap& properties);
};

/// Runner plus the GgmlModel behind it. The caller registers `model` as the weight base for the
/// artifact's (GGML_WEIGHTS, GGML_BACKEND) pair when it uploaded the weights itself.
struct RunnerResult {
    std::shared_ptr<Runner> runner;
    std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> model;
};

/// Picks the runner from the graph inputs the artifact declares. `base`, when non-null, is the
/// already-uploaded model of the same GGUF: the new graph shares its backend and weights and
/// keeps its own KV caches (CacheMode::Fresh).
RunnerResult make_runner(const std::filesystem::path& artifact, const RunnerConfig& config,
                         const std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>& base = nullptr);

}  // namespace ggml_plugin
}  // namespace ov
