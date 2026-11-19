// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <mutex>

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"
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

/// One compiled artifact. Holds the shared weight upload (see Plugin) and a template runner that
/// provides the model signature and properties; every infer request builds a runner of its own on
/// top of the upload (see InferRequest), so requests never share a KV cache, block pool or
/// recurrent state. The template's own KV caches are allocated but never used -- one set per
/// compiled model, the price of deriving the signature from a real runner.
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(RunnerResult result, std::filesystem::path artifact, RunnerConfig config,
                  const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    const std::filesystem::path& artifact() const {
        return m_artifact;
    }
    const RunnerConfig& config() const {
        return m_config;
    }
    const std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel>& template_model() const {
        return m_template;
    }
    /// Per-request runners are built over the upload; refuse loudly if the template kept none
    /// (e.g. the registry entry expired between two compiles).
    bool shares_upload() const {
        return m_template != nullptr;
    }

private:
    std::shared_ptr<Runner> m_template_runner;  // signature and properties only; never infers
    std::shared_ptr<const ov::Model> m_signature;
    std::filesystem::path m_artifact;
    RunnerConfig m_config;
    std::shared_ptr<ov::ggml_cgraph_loader::GgmlModel> m_template;
};

/// Each request runs its own runner over the compiled model's shared weight upload: its own KV
/// caches, block pool (paged) and recurrent state, so several pipelines on one compiled model do
/// not collide. The runner is built in the constructor: GenAI creates exactly one request per
/// pipeline, so the build (a JSON parse, host graph rebuild and a device allocation for the KV
/// caches) is paid once per pipeline, not per generate().
///
/// All requests execute on the compiled model's one ggml backend; running requests on different
/// threads at the same time is not supported -- infer them one after the other. The model
/// advertises one optimal request.
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
