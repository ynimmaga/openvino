// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <filesystem>

#include "openvino/core/model.hpp"
#include "openvino/ggml_cgraph_loader/ggml_model.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace ggml_plugin {

namespace {

/// Stands for the whole ggml graph in the compiled model's runtime model. It only carries the
/// port types and shapes; execution goes through the Runner.
class GgmlGraph : public ov::op::Op {
public:
    OPENVINO_OP("GgmlGraph", "ggml");

    GgmlGraph(const ov::OutputVector& args, std::vector<PortDesc> outputs) : Op(args), m_outputs(std::move(outputs)) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        for (size_t i = 0; i < m_outputs.size(); i++) {
            set_output_type(i, m_outputs[i].type, m_outputs[i].shape);
        }
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<GgmlGraph>(new_args, m_outputs);
    }

private:
    std::vector<PortDesc> m_outputs;
};

std::shared_ptr<const ov::Model> make_signature(const Runner& runner) {
    ov::ParameterVector params;
    ov::OutputVector args;
    for (const auto& in : runner.inputs()) {
        auto p = std::make_shared<ov::op::v0::Parameter>(in.type, in.shape);
        p->set_friendly_name(in.name);
        p->output(0).set_names({in.name});
        params.push_back(p);
        args.push_back(p);
    }
    const auto outs = runner.outputs();
    auto graph = std::make_shared<GgmlGraph>(args, outs);
    ov::ResultVector results;
    for (size_t i = 0; i < outs.size(); i++) {
        graph->output(i).set_names({outs[i].name});
        results.push_back(std::make_shared<ov::op::v0::Result>(graph->output(i)));
    }
    return std::make_shared<ov::Model>(results, params, "ggml");
}

ov::Shape empty_shape(const ov::PartialShape& shape) {
    return ov::Shape(shape.rank().is_static() ? shape.rank().get_length() : 1, 0);
}

}  // namespace

// ---------------------------------------------------------------------------------------------- Plugin

Plugin::Plugin() {
    set_device_name("GGML");
}

namespace {
// The same GGUF on the same backend gets one upload; a different spelling of the path or an
// empty backend name (auto) means a separate entry, which is conservative, never wrong.
std::string weights_key(const ov::AnyMap& config) {
    const auto w = config.find("GGML_WEIGHTS");
    if (w == config.end()) {
        return {};
    }
    std::error_code ec;
    std::filesystem::path p = std::filesystem::weakly_canonical(
        std::filesystem::path(w->second.as<std::string>()), ec);
    if (ec) {
        p = std::filesystem::absolute(w->second.as<std::string>());
    }
    std::string backend;
    if (const auto b = config.find("GGML_BACKEND"); b != config.end()) {
        backend = b->second.as<std::string>();
    }
    return p.string() + "|" + backend;
}
}  // namespace

std::shared_ptr<ggml_cgraph_loader::GgmlModel> Plugin::weights_base(const ov::AnyMap& config) const {
    const std::string key = weights_key(config);
    if (key.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(m_weights_mutex);
    const auto it = m_weights.find(key);
    if (it == m_weights.end()) {
        return nullptr;
    }
    if (auto base = it->second.lock()) {
        return base;
    }
    m_weights.erase(it);  // expired: the compiled models that owned it are gone
    return nullptr;
}

void Plugin::remember_weights(const ov::AnyMap& config,
                              const std::shared_ptr<ggml_cgraph_loader::GgmlModel>& model) const {
    const std::string key = weights_key(config);
    if (key.empty() || !model) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_weights_mutex);
    auto& entry = m_weights[key];
    if (entry.lock()) {
        return;  // another thread uploaded first; its base stays
    }
    entry = model;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::filesystem::path& model_path,
                                                          const ov::AnyMap& properties) const {
    ov::AnyMap config = m_config;
    for (const auto& [k, v] : properties) {
        config[k] = v;
    }
    const RunnerConfig cfg = RunnerConfig::parse(config);
    const auto base = weights_base(config);
    auto result = make_runner(model_path, cfg, base);
    if (!base) {
        remember_weights(config, result.model);
    }
    return std::make_shared<CompiledModel>(std::move(result), model_path, cfg, shared_from_this());
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>&,
                                                          const ov::AnyMap&) const {
    OPENVINO_THROW("[GGML] the GGML device compiles cgraph artifacts by path: "
                   "core.compile_model(\"<artifact>.json\", \"GGML\", {{\"GGML_WEIGHTS\", \"<model>.gguf\"}})");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties,
                                                          const ov::SoPtr<ov::IRemoteContext>&) const {
    return compile_model(model, properties);
}

void Plugin::set_property(const ov::AnyMap& properties) {
    for (const auto& [k, v] : properties) {
        m_config[k] = v;
    }
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap&) const {
    if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{ov::supported_properties,
                                             ov::available_devices,
                                             ov::device::full_name,
                                             ov::device::capabilities};
    } else if (name == ov::internal::supported_properties.name()) {
        return std::vector<ov::PropertyName>{};
    } else if (name == ov::available_devices.name()) {
        return std::vector<std::string>{""};
    } else if (name == ov::device::full_name.name()) {
        return std::string("ggml");
    } else if (name == ov::device::capabilities.name()) {
        return std::vector<std::string>{};
    }
    if (auto it = m_config.find(name); it != m_config.end()) {
        return it->second;
    }
    OPENVINO_THROW("[GGML] unsupported property ", name);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream&, const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream&,
                                                         const ov::SoPtr<ov::IRemoteContext>&,
                                                         const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor&, const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor&,
                                                         const ov::SoPtr<ov::IRemoteContext>&,
                                                         const ov::AnyMap&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const {
    return {};
}

// --------------------------------------------------------------------------------------- CompiledModel

CompiledModel::CompiledModel(RunnerResult result, std::filesystem::path artifact, RunnerConfig config,
                             const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::ICompiledModel(make_signature(*result.runner), plugin),
      m_template_runner(std::move(result.runner)),
      m_signature(make_signature(*m_template_runner)),
      m_artifact(std::move(artifact)),
      m_config(std::move(config)),
      m_template(std::move(result.model)) {}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

void CompiledModel::export_model(std::ostream&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    return m_signature;
}

void CompiledModel::set_property(const ov::AnyMap&) {}

ov::Any CompiledModel::get_property(const std::string& name) const {
    const auto runner_properties = m_template_runner->properties();
    if (name == ov::supported_properties.name()) {
        std::vector<ov::PropertyName> names{ov::supported_properties,
                                            ov::model_name,
                                            ov::execution_devices,
                                            ov::loaded_from_cache,
                                            ov::optimal_number_of_infer_requests,
                                            ov::PropertyName("GGML_BACKEND")};
        for (const auto& [key, value] : runner_properties) {
            names.emplace_back(key);
        }
        return names;
    } else if (auto it = runner_properties.find(name); it != runner_properties.end()) {
        return it->second;
    } else if (name == ov::model_name.name()) {
        return std::string("ggml");
    } else if (name == ov::execution_devices.name()) {
        return std::vector<std::string>{get_plugin()->get_device_name()};
    } else if (name == ov::loaded_from_cache.name()) {
        return false;
    } else if (name == ov::optimal_number_of_infer_requests.name()) {
        return 1u;
    } else if (name == "GGML_BACKEND") {
        return m_template_runner->backend_name();
    }
    OPENVINO_THROW("[GGML] unsupported compiled model property ", name);
}

// ---------------------------------------------------------------------------------------- InferRequest

InferRequest::InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    OPENVINO_ASSERT(compiled_model->shares_upload(),
                    "[GGML] the compiled model kept no weight upload to build a per-request runner over");
    m_runner = make_runner(compiled_model->artifact(), compiled_model->config(), compiled_model->template_model()).runner;
    for (const auto& port : get_inputs()) {
        allocate_tensor(port, [&port](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(port.get_element_type(), empty_shape(port.get_partial_shape()));
        });
    }
    for (const auto& port : get_outputs()) {
        allocate_tensor(port, [&port](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(port.get_element_type(), empty_shape(port.get_partial_shape()));
        });
    }
}

void InferRequest::infer() {
    TensorMap inputs, outputs;
    for (const auto& port : get_inputs()) {
        inputs[port.get_any_name()] = get_tensor(port);
    }
    for (const auto& port : get_outputs()) {
        outputs[port.get_any_name()] = get_tensor(port);
    }
    m_runner->infer(inputs, outputs);
}

std::vector<ov::SoPtr<ov::IVariableState>> InferRequest::query_state() const {
    return m_runner->states();
}

std::vector<ov::ProfilingInfo> InferRequest::get_profiling_info() const {
    return {};
}

}  // namespace ggml_plugin
}  // namespace ov

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_ggml_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::ggml_plugin::Plugin, version)
