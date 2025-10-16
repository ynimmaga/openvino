// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <format>
#include <regex>

#include "ov_compute.h"
#include "openvino/frontend/manager.hpp"
#include "ov_ep_context.h"
#include "ov_utils.h"

using namespace onnxruntime::openvino_ep_plugin;

namespace onnxruntime {
namespace openvino_ep_plugin {

OvComputeInfo::OvComputeInfo(ApiPtrs apis, ov::Core& ov_core, Ort::Logger logger) : ApiPtrs(apis), ov_core_(ov_core), logger_(logger) {
  ort_version_supported = ORT_API_VERSION;
  OrtNodeComputeInfo::CreateState = CreateStateImpl;
  OrtNodeComputeInfo::Compute = ComputeImpl;
  OrtNodeComputeInfo::ReleaseState = ReleaseStateImpl;
}

std::shared_ptr<ov::Model> OvComputeInfo::ReadModelWithExternalWeightSupport(std::string model, const std::filesystem::path& model_path) {
  std::istringstream modelStringStream(std::move(model));
  std::istream& modelStream = modelStringStream;
  // Try to load with FrontEndManager
  ov::frontend::FrontEndManager manager;
  ov::frontend::FrontEnd::Ptr FE;
  ov::frontend::InputModel::Ptr inputModel;

  ov::AnyVector params{&modelStream, model_path.string()};

  FE = manager.load_by_model(params);
  OVEP_ENFORCE(FE, "Failed to load FrontEnd");
  inputModel = FE->load(params);
  return FE->convert(inputModel);
}

bool static ShouldRecompile(const char* ov_exception) noexcept {
  try {
    std::string ov_exception_string(ov_exception);
    std::smatch matches;
    std::regex error_message_pattern(R"(\bZE_\w*\b)");
    if (!std::regex_search(ov_exception_string, matches, error_message_pattern)) {
      return false;
    }
    std::regex error_code_pattern("code 0x([0-9a-fA-F]+)");
    uint32_t error_code{0};
    if (std::regex_search(ov_exception_string, matches, error_code_pattern)) {
      std::from_chars(&(*matches[1].first), &(*matches[1].second), error_code, 16);
      return error_code == 0x7800000f /* ZE_RESULT_ERROR_INVALID_NATIVE_BINARY */;
    }
  } catch (...) {
    // don't want to throw since we intend to call from an exception handler already.
  }
  return false;
}

OrtStatus* OvComputeInfo::Init(const std::string& ov_device, const ov::AnyMap& configs, const OvModelInfo& model_info, EpContextNode ep_context_node, std::vector<ModelTransformation> transformations) {
  const OnnxIOMapping& io_mapping = model_info.io_mapping;
  switch (ep_context_node.private_fields_.type) {
    case EpContextNode::EpContextType::Native:
      if (!transformations.empty()) {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Model is native ep ctx. Model transformation functions will be skipped.");
      }

      try {
        if (ep_context_node.embed_mode != 0) {
          std::istringstream model_stream(std::move(ep_context_node.ep_cache_context));
          compiled_model_ = ov_core_.import_model(model_stream, ov_device, configs);
        } else {
          const std::filesystem::path ep_ctx_path = ep_context_node.private_fields_.epctx_dir / ep_context_node.ep_cache_context;
#if (OPENVINO_VERSION_MAJOR > 2025 || (OPENVINO_VERSION_MAJOR == 2025 && OPENVINO_VERSION_MINOR >= 3))
          compiled_model_ = ov_core_.import_model(ov::read_tensor_data(ep_ctx_path), ov_device, configs);
#else
          std::ifstream model_stream(ep_ctx_path, std::ios_base::binary | std::ios_base::in);
          compiled_model_ = ov_core_.import_model(model_stream, ov_device, configs);
#endif
        }
      } catch (ov::Exception& e) {
        if (ShouldRecompile(e.what())) {
          return ort_api.CreateStatus(ORT_MODEL_REQUIRES_COMPILATION, OVEP_ERROR_STR(e.what()).c_str());
        }
        return ort_api.CreateStatus(ORT_INVALID_GRAPH, OVEP_ERROR_STR(e.what()).c_str());
      }
      break;

    case EpContextNode::EpContextType::OV_IR:
      if (ep_context_node.embed_mode != 0) {
        // To support this we would need to save the weights location somewhere, or have a schema for embedding the weights along with the IR.
        return Ort::Status("Epctx with OVIR must use embed_mode == 0", ORT_INVALID_ARGUMENT).release();
      } else {
        const std::filesystem::path ep_ctx_path = ep_context_node.private_fields_.epctx_dir / ep_context_node.ep_cache_context;
        if (transformations.empty()) {
          compiled_model_ = ov_core_.compile_model(ep_ctx_path, ov_device, configs);
        } else {
          const std::filesystem::path bin_path = ep_ctx_path.parent_path() / (ep_ctx_path.stem().string() + ".bin");
          std::shared_ptr<ov::Model> ov_model = ov_core_.read_model(ep_ctx_path, bin_path);
          for (const auto& transform : transformations) {
            OVEP_ENFORCE(transform.transform_func, "Transform function is null");
            OVEP_RETURN_IF_ERROR(transform.transform_func(ov_model));
          }
          compiled_model_ = ov_core_.compile_model(ov_model, ov_device, configs);
        }
      }
      break;

    default:
      return Ort::Status("Unsupported EpContextType", ORT_INVALID_ARGUMENT).release();
  }

  InitCommon(io_mapping, std::move(transformations));
  return nullptr;
}

OrtStatus* OvComputeInfo::Init(const std::string& ov_device, const ov::AnyMap& configs, const OvModelInfo& model_info, std::string model, std::vector<ModelTransformation> transformations) {
  const OnnxIOMapping& io_mapping = model_info.io_mapping;
  if (transformations.empty() && !model_info.has_external_weights) {
    compiled_model_ = ov_core_.compile_model(std::move(model), {}, ov_device, configs);
  } else {
    std::shared_ptr<ov::Model> ov_model = ReadModelWithExternalWeightSupport(std::move(model), model_info.model_path);
    for (const auto& transform : transformations) {
      OVEP_ENFORCE(transform.transform_func, "Transform function is null");
      OVEP_RETURN_IF_ERROR(transform.transform_func(ov_model));
    }
    compiled_model_ = ov_core_.compile_model(ov_model, ov_device, configs);
  }

  InitCommon(io_mapping, std::move(transformations));
  return nullptr;
}

void OvComputeInfo::InitCommon(const OnnxIOMapping& io_mapping, std::vector<ModelTransformation> transformations) {
  auto initializers = [transformations = std::move(transformations)](InferRequestPool::OVInferRequestPtr& infer_request) {
    for (const auto& transform : transformations) {
      if (transform.infer_request_initializer) {
        transform.infer_request_initializer(infer_request->ov());
      }
    }
  };

  infer_request_pool_ = std::make_unique<InferRequestPool>(compiled_model_, 1, std::move(initializers));
  onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(compiled_model_, io_mapping, SessionContext{});
}

OrtStatus* OvComputeInfo::Export(EpContextNode& epctx_node) {
  if (epctx_node.embed_mode) {
    std::stringstream ss;
    compiled_model_.export_model(ss);
    epctx_node.ep_cache_context = std::move(ss).str();
  } else {
    std::ofstream output_file(epctx_node.private_fields_.epctx_dir / epctx_node.ep_cache_context, std::ios_base::binary);
    compiled_model_.export_model(output_file);
  }
  return nullptr;
}

OrtStatus* OvComputeInfo::Compute(void* /*compute_state*/,
                                  OrtKernelContext* kernel_context) {
  auto guarded_infer_req = infer_request_pool_->getRequest();
  auto& infer_request = guarded_infer_req.infer_request_;

  Ort::KernelContext context(kernel_context);

  if (onnx_to_ov_bindings_->has_dynamic_io_) {
    // Dynamic shape inference

    // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
    // tensors instead of binding them to the infer request directly.

    // Bind inputs
    for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
      // Set the input shape based on the input tensor from ort
      auto tensor = context.GetInput(input_info.onnx_index);
      auto&& ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
      auto input_shape = ParameterShape(ort_shape);

      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_shape,
                               const_cast<void*>(tensor.GetTensorRawData()));
    }

    // Run Inference
    infer_request->Infer();

    // Copy outputs
    for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
      auto ov_tensor = infer_request->ov().get_tensor(output_info.name);
      auto&& output_shape = ParameterShape::ToOrtShape(ov_tensor.get_shape());
      auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);

      OVEP_RETURN_IF(ov_tensor.get_byte_size() != ort_tensor.GetTensorSizeInBytes(),
                     ort_api,
                     std::format("Output tensor size mismatch for {}", output_info.name).c_str());

      std::memcpy(ort_tensor.GetTensorMutableRawData(),
                  ov_tensor.data(),
                  ov_tensor.get_byte_size());
    }
  } else {
    // Static shape inference

    // Bind inputs
    for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_info.shape,
                               const_cast<void*>(context.GetInput(input_info.onnx_index).GetTensorRawData()));
    }

    // Bind outputs
    for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
      infer_request->SetTensor(output_info.name,
                               output_info.type,
                               output_info.shape,
                               context.GetOutput(output_info.onnx_index, output_info.shape).GetTensorMutableRawData());
    }

    // Run Inference
    infer_request->Infer();
  }

  return nullptr;
}

OrtStatus* OvComputeInfo::SetWorkloadType(const std::string& workload_type) {
  if (compiled_model_) {
    try {
      compiled_model_.set_property(ov::workload_type(workload_type));
    } catch (const std::exception& e) {
      return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION,
                                  std::format("set_property(ov::workload_type(%s)) failed. Details: %s", workload_type, e.what()).c_str());
    }
  } else {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "No compiled model to set workload_type on");
  }
  return nullptr;
}

OrtStatus* OvComputeInfo::CreateState(OrtNodeComputeContext* compute_context,
                                      void** compute_state) {
  // Dummy implementation: set compute_state to nullptr
  (void)compute_context;
  *compute_state = nullptr;
  return nullptr;
}

void OvComputeInfo::ReleaseState(void* compute_state) {
  // Dummy implementation: do nothing
  (void)compute_state;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
