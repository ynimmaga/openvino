// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <mutex>

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "runner.hpp"

namespace ov {
namespace ggml_cgraph_loader {
class GgmlModel;
}  // namespace ggml_cgraph_loader

namespace ggml_plugin {

/// "GGML" device: executes cgraph artifacts (ggml-cgraph-generator) on a ggml backend.
///
///   core.compile_model("decoder_b1.json", "GGML", {{"GGML_WEIGHTS", "model.gguf"}});
///
/// Models are compiled from an artifact path only; there is no ov::Model to compile, because the
/// artifact is a ggml graph rather than an OpenVINO one.
///
/// Compiling several artifacts over one GGUF (GGML_WEIGHTS) uploads the weights once: the first
/// compiled model becomes the base for the rest, which share its backend and weight buffer and
/// keep their own KV caches. The key includes the backend, so a different GGML_BACKEND gets its
/// own upload.
class Plugin : public ov::IPlugin {
public:
    Plugin();

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::filesystem::path& model_path,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    /// Already-uploaded model for a (GGML_WEIGHTS, GGML_BACKEND) pair, if one is alive.
    std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> weights_base(const ov::AnyMap& config) const;
    /// Record `model` as the uploaded base for its (GGML_WEIGHTS, GGML_BACKEND) pair.
    void remember_weights(const ov::AnyMap& config,
                          const std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>& model) const;

    ov::AnyMap m_config;
    /// Weak: when the last compiled model built over a base is destroyed, the upload (its backend
    /// and weight buffer) is freed and the entry expires on the next compile.
    mutable std::mutex m_weights_mutex;
    mutable std::map<std::string, std::weak_ptr<ov::ggml_cgraph_loader::GgmlModel>> m_weights;
};

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(std::shared_ptr<Runner> runner, const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    const std::shared_ptr<Runner>& runner() const {
        return m_runner;
    }

private:
    CompiledModel(std::shared_ptr<Runner> runner,
                  std::shared_ptr<const ov::Model> signature,
                  const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<Runner> m_runner;
    std::shared_ptr<const ov::Model> m_signature;
};

/// All requests of one compiled model share its runner: the weights and the KV cache live in one
/// ggml graph, so the model advertises a single optimal request.
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model);

    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    std::shared_ptr<Runner> m_runner;
};

}  // namespace ggml_plugin
}  // namespace ov
