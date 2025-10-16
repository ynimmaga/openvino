// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <string>
#include <set>
#include <unordered_set>
#include <filesystem>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "ov_utils.h"
#include "ov_plugin_ver.h"

#include "openvino/runtime/core.hpp"
#include "ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

struct EpContextSessionConfig {
  bool enable_{false};
  bool embed_{false};
  bool share_{false};
  std::filesystem::path path_{};
};

struct EpSessionOptions {
  bool cpu_fallback_disabled{false};
};

using ConfigMap = std::map<std::string, ov::AnyMap>;
using ReshapeMap = std::map<std::string, ov::PartialShape>;

struct OpenVINOEpProviderOptions {
  ConfigMap load_config_{};      // JSON config map to load custom OV parameters.
  bool enable_causallm_{false};  // Enables Causal LM Compilation for ORT GenAI OVEP Pass
  ReshapeMap reshape_input_{};   // Used for reshaping the OV graph inputs shape at runtime.

  static const std::unordered_set<std::string>& GetValidProviderKeys() {
    static const std::unordered_set<std::string> valid_keys = {
        "load_config", "enable_causallm", "reshape_input"};
    return valid_keys;
  }

  auto GetDeviceConfig(const std::string& ov_device) const {
    std::string device_name = ov_device;

    // Get meta device name, For eg AUTO when ov_device is AUTO:CPU,GPU
    auto pos = ov_device.find_first_of(':');
    if (pos != std::string::npos) {
      device_name = ov_device.substr(0, pos);
    }

    auto itr = load_config_.find(device_name);
    return itr != load_config_.end() ? itr->second : ov::AnyMap{};
  }
};

struct OpenVINOEpPluginOptions {
  OpenVINOEpPluginOptions() = default;
  OpenVINOEpPluginOptions(std::string ep_name) : ep_name_(std::move(ep_name)) {}
  EpContextSessionConfig ep_ctx_;
  EpSessionOptions ep_session_options_;
  OpenVINOEpProviderOptions provider_options_;
  OrtStatus* Init(const OrtApi& ort_api, const Ort::Logger& logger, const OrtSessionOptions& session_options, const std::string& ep_name);

 private:
  std::string ep_name_;
  OrtStatus* ParseProviderOptions(const OrtApi& ort_api, const Ort::Logger& logger, const OrtSessionOptions& session_options, const std::string& ep_name);
  OrtStatus* ParseEpContextOptions(const OrtApi& ort_api, const OrtSessionOptions& session_options);
  OrtStatus* ParseEpSessionOptions(const OrtApi& ort_api, const OrtSessionOptions& session_options);
};

struct ModelTransformation {
  using InferRequestInitialzer = typename std::function<void(ov::InferRequest&)>;
  using TransformFunc = typename std::function<OrtStatus*(std::shared_ptr<ov::Model>&)>;

  TransformFunc transform_func;
  InferRequestInitialzer infer_request_initializer;  // optional will be checked before use

  ModelTransformation(TransformFunc func, InferRequestInitialzer initializer = nullptr)
      : transform_func(std::move(func)), infer_request_initializer(std::move(initializer)) {}
};

struct OvComputeInfo;
struct OnnxIOMapping;
class OpenVINOEpPlugin : public OrtEp,
                         public ApiPtrs {
 public:
  OpenVINOEpPlugin(ApiPtrs apis, const std::string& name, const OpenVINOEpPluginOptions& options, const OrtLogger& logger, const std::string ov_device_type, std::shared_ptr<ov::Core> ov_core);
  ~OpenVINOEpPlugin();

  OVEP_DISABLE_COPY_AND_MOVE(OpenVINOEpPlugin)

  // Member functions that implement the OpenVINO EP functionality
  const char* GetName() const noexcept {
    return name_.c_str();
  }
  OrtStatus* GetCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info);
  OrtStatus* Compile(const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes);
  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info, OrtAllocator** allocator);
  void ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos, size_t num_node_compute_infos);

  std::vector<ModelTransformation> BuildTransformationPipeline(std::filesystem::path model_dir);
  OrtStatus* SetDynamicOptions(const char* const* option_keys, const char* const* option_values, size_t num_options);

  struct ModelFilePaths {
    std::filesystem::path model_path;
    std::filesystem::path output_model_path;
  };
  OrtStatus* GetFilePaths(const OrtGraph* graph, ModelFilePaths& paths) const;

 public:
  // Static wrapper functions for C API compatibility
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept {
    const auto* ep = static_cast<const OpenVINOEpPlugin*>(this_ptr);
    // No exception expected
    return ep->GetName();
  }

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    return ApiEntry([&]() { return ep->GetCapability(graph, graph_support_info); }, ep->logger_);
  }

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                             size_t count, OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** ep_context_nodes) noexcept {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    return ApiEntry([&]() { return ep->Compile(graphs, fused_nodes, count, node_compute_infos, ep_context_nodes); }, ep->logger_);
  }

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    ApiEntry([&]() { ep->ReleaseNodeComputeInfos(node_compute_infos, num_node_compute_infos); }, ep->logger_);
  }

  static OrtStatus* ORT_API_CALL SetDynamicOptionsImpl(OrtEp* this_ptr, const char* const* option_keys,
                                                       const char* const* option_values, size_t num_options) noexcept {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    return ApiEntry([&]() { return ep->SetDynamicOptions(option_keys, option_values, num_options); }, ep->logger_);
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEp* this_ptr, const OrtMemoryInfo* memory_info, OrtAllocator** allocator) noexcept {
    auto* ep = static_cast<OpenVINOEpPlugin*>(this_ptr);
    return ApiEntry([&]() { return ep->CreateAllocator(memory_info, allocator); }, ep->logger_);
  }

  static constexpr const char* provider_name_ = "OpenVINOExecutionProvider";

 private:
  std::string name_;
  Ort::Logger logger_;
  std::string ov_device_type_;  // OpenVINO device type (CPU, GPU, NPU, AUTO, etc.)
  std::shared_ptr<ov::Core> ov_core_;
  std::shared_ptr<SharedContext> shared_context_;
  const OpenVINOEpPluginOptions options_;
  std::set<OvComputeInfo*> computes_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
