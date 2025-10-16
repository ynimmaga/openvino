// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

#include "ov_compute_stateful.h"
#include "ov_ep_context.h"
#include "../common/ov_stateful_patch_utils.h"

using namespace onnxruntime::openvino_ep;

#define STATEFUL_LOG_LEVEL ORT_LOGGING_LEVEL_INFO

namespace onnxruntime {
namespace openvino_ep_plugin {

OvComputeInfoStateful::OvComputeInfoStateful(ApiPtrs apis, ov::Core& ov_core, const Ort::Logger& logger) : OvComputeInfo(apis, ov_core, logger) {}

OrtStatus* OvComputeInfoStateful::Init(const std::string& ov_device, const ov::AnyMap& configs, const OvModelInfo& model_info, EpContextNode ep_context_node, std::vector<ModelTransformation>) {
  const OnnxIOMapping& io_mapping = model_info.io_mapping;
  switch (ep_context_node.private_fields_.type) {
    case EpContextNode::EpContextType::Native:
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
      break;
    case EpContextNode::EpContextType::OV_IR:
      // Outstanding open to MSFT: How do I get the path to the graph? Graph API doesn't have the source model path.
      if (ep_context_node.embed_mode != 0) {
        // To support this we would need to save the weights location somewhere, or have a schema for embedding the weights along with the IR.
        return Ort::Status("Epctx with OVIR must use embed_mode == 0", ORT_INVALID_ARGUMENT).release();
      } else {
        const std::filesystem::path ep_ctx_path = ep_context_node.private_fields_.epctx_dir / ep_context_node.ep_cache_context;
        auto model = ov_core_.read_model(ep_ctx_path);
        OVEP_RETURN_IF_ERROR(StatefulCompile(model, ov_device, configs));
      }
      break;
    default:
      return Ort::Status("Unsupported EpContextType", ORT_INVALID_ARGUMENT).release();
  }

  return StatefulPostCompileInit(ov_device, io_mapping);
}

OrtStatus* OvComputeInfoStateful::Init(const std::string& ov_device, const ov::AnyMap& configs, const OvModelInfo& model_info, std::string model, std::vector<ModelTransformation>) {
  const OnnxIOMapping& io_mapping = model_info.io_mapping;
  auto ov_model = ReadModelWithExternalWeightSupport(std::move(model), model_info.model_path);
  OVEP_RETURN_IF_ERROR(StatefulCompile(ov_model, ov_device, configs));

  return StatefulPostCompileInit(ov_device, io_mapping);
}

OrtStatus* OvComputeInfoStateful::StatefulPostCompileInit(const std::string& ov_device, const OnnxIOMapping& io_mapping) {
  auto has_tensor = [&](const std::string& tensor_name) -> bool {
    for (const auto& input : compiled_model_.inputs()) {
      const auto& names = input.get_names();
      if (names.find(tensor_name) != names.end()) {
        return true;
      }
    }
    return false;
  };

  auto tensor_has_type = [&](const std::string& tensor_name, const ov::element::Type& type) -> bool {
    const auto& input = compiled_model_.input(tensor_name);
    return input.get_element_type() == type;
  };

  if (!has_tensor("beam_idx") || !tensor_has_type("beam_idx", ov::element::i32)) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Compiled Model must have 'beam_idx' tensor of element type i32");
  }

  if (!has_tensor("input_ids") || !tensor_has_type("input_ids", ov::element::i64)) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "Compiled Model must have 'input_ids' tensor of element type i64");
  }

  has_position_ids_ = has_tensor("position_ids");
  if (has_position_ids_ && !tensor_has_type("position_ids", ov::element::i64)) {
    return ort_api.CreateStatus(ORT_RUNTIME_EXCEPTION, "'position_ids' tensor must be of type i64");
  }

  _infer_request = std::make_unique<WrappedInferRequest>(std::move(compiled_model_.create_infer_request()));
  onnx_to_ov_bindings_ = std::make_unique<OnnxToOvNetworkBindings>(compiled_model_, io_mapping, SessionContext{true});

  // For NPU & GPU, use full chat history for each pre-fill.
  // For GPU, we do this right now as rewinding kvcache takes a really long time -- which we need to address.
  prefill_use_full_chat_history_ = ((ov_device.find("NPU") != std::string::npos) || (ov_device.find("GPU") != std::string::npos));

  return nullptr;
}

static void LogBasicModelInfo(const std::shared_ptr<const ov::Model>& model, const Ort::Logger& logger) {
  if (STATEFUL_LOG_LEVEL < logger.GetLoggingSeverityLevel()) {
    return;
  }
  ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "Model Name: %s", model->get_friendly_name().c_str());
  // Log detailed information about model inputs and outputs
  auto inputs = model->inputs();
  auto outputs = model->outputs();
  ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\tInputs: ");
  for (const ov::Output<const ov::Node>& input : inputs) {
    const std::string name = input.get_any_name();
    const ov::element::Type type = input.get_element_type();
    const ov::PartialShape shape = input.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(input);
    ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\t\t%s, %s, %s, %s", name.c_str(), type.to_string().c_str(), shape.to_string().c_str(), layout.to_string().c_str());
  }
  ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\tOutputs: ");
  for (const ov::Output<const ov::Node>& output : outputs) {
    const std::string name = output.get_any_name();
    const ov::element::Type type = output.get_element_type();
    const ov::PartialShape shape = output.get_partial_shape();
    const ov::Layout layout = ov::layout::get_layout(output);
    ORT_CXX_LOGF(logger, STATEFUL_LOG_LEVEL, "\t\t%s, %s, %s, %s", name.c_str(), type.to_string().c_str(), shape.to_string().c_str(), layout.to_string().c_str());
  }
}

OrtStatus* OvComputeInfoStateful::StatefulCompile(std::shared_ptr<ov::Model> model, const std::string& ov_device, const ov::AnyMap& device_config) {
  ov::AnyMap config = device_config;

  ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "Model input to Stateful Compile:");
  LogBasicModelInfo(model, logger_);

  bool is_stateful = IsStateful(model);
  ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "Model IsStateful(): %s", is_stateful ? "True" : "False");

  if (!is_stateful) {
    PatchStatefulDecoder(model);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "Model after stateful transformation:");
    LogBasicModelInfo(model, logger_);
  }

  auto kv_pos = GetKVAxesPos(model);
  if (ov_device.find("NPU") != std::string::npos) {
    KVDesc kv_desc;
    kv_desc.max_prompt_len = PopIntAndCast(config, "MAX_PROMPT_LEN").value_or(1024u);
    kv_desc.min_response_len = PopIntAndCast(config, "MIN_RESPONSE_LEN").value_or(128u);

    // For compilation, MAX_PROMPT_LEN & MIN_RESPONSE_LEN should not be 0
    if (kv_desc.max_prompt_len == 0 || kv_desc.min_response_len == 0) {
      throw std::runtime_error("MAX_PROMPT_LEN or MIN_RESPONSE_LEN cannot be 0");
    }

    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.batch = %zu", kv_pos.batch);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.seq_len = %zu", kv_pos.seq_len);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.max_prompt_len = %u", kv_desc.max_prompt_len);
    ORT_CXX_LOGF(logger_, STATEFUL_LOG_LEVEL, "kv_pos.min_response_len = %u", kv_desc.min_response_len);
    UpdateNPUConfig(config, kv_pos, kv_desc);
  } else {
    // This patches the OV IR model so that it only produces the logits required for sampling.
    // It's not needed for NPU, as it is already internally applied inside NPUW compilation.
    ApplySliceBeforeMatmulTransformation(model);
  }

  compiled_model_ = ov_core_.compile_model(model, ov_device, config);

  return nullptr;
}

static inline void FillTensor(ov::InferRequest infer_request, const std::string& tensor_name, const ov::element::Type& type,
                              const std::vector<size_t>& shape, int32_t fill_value) {
  ov::Tensor tensor = ov::Tensor(type, shape);
  std::fill_n(tensor.data<int32_t>(), tensor.get_size(), fill_value);
  infer_request.set_tensor(tensor_name, tensor);
}

static inline void CacheTensor(ov::InferRequest infer_request, const std::string& tensor_name, std::vector<int64_t>& cache) {
  auto tensor = infer_request.get_tensor(tensor_name);
  auto* pData = tensor.data<int64_t>();
  for (size_t i = 0; i < tensor.get_size(); i++) {
    cache.emplace_back(pData[i]);
  }
}

static inline void SetTensorFromCache(ov::InferRequest infer_request, const std::string& tensor_name,
                                      const std::vector<int64_t>& cache_data) {
  auto tensor = infer_request.get_tensor(tensor_name);
  auto new_shape = tensor.get_shape();
  new_shape[1] = cache_data.size();

  auto new_tensor = ov::Tensor(tensor.get_element_type(), new_shape);
  auto* pNewData = new_tensor.data<int64_t>();
  std::memcpy(pNewData, cache_data.data(), cache_data.size() * sizeof(int64_t));

  infer_request.set_tensor(tensor_name, new_tensor);
}

OrtStatus* OvComputeInfoStateful::PreInfer() {
  auto& infer_request = _infer_request->ov();

  // Workaround: Setting the value here as it cannot be set at the ORT GenAI layer currently.
  // TODO(ankit): Address this issue and implement the fix at the appropriate layer.
  FillTensor(infer_request, "beam_idx", ov::element::i32, {1}, 0);

  // If 'prefill use full chat history' mode is enabled, we need to cache input_ids and position_ids.
  if (prefill_use_full_chat_history_) {
    auto input_ids_tensor = infer_request.get_tensor("input_ids");
    CacheTensor(infer_request, "input_ids", cached_input_ids_);

    // "position_ids" (GQA with Rotary Embeddings doesnt have position_ids) - check if exists
    if (has_position_ids_) {
      CacheTensor(infer_request, "position_ids", cached_position_ids_);
    }

    // If we're about to run the prefill model
    if (input_ids_tensor.get_size() > 1) {
      // Check if the size of the current "input_ids" tensor does not match the size of the cached "input_ids".
      // This indicates that we are running a subsequent prompt (not the initial prefill).
      if (input_ids_tensor.get_shape()[1] != cached_input_ids_.size()) {
        // Clear the internal KVCache state. For NPU device, this operation is a no-op.
        infer_request.reset_state();

        // Set tensors using cached values
        SetTensorFromCache(infer_request, "input_ids", cached_input_ids_);

        // Only override position_ids if it exists and we have cached values
        if (has_position_ids_ && !cached_position_ids_.empty()) {
          SetTensorFromCache(infer_request, "position_ids", cached_position_ids_);
        }
      }
    }
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::Compute(void* /*compute_state*/,
                                          OrtKernelContext* kernel_context) {
  Ort::KernelContext context(kernel_context);

  if (!onnx_to_ov_bindings_->has_dynamic_io_) {
    return Ort::Status("Stateful Compute requires dynamic IO", ORT_RUNTIME_EXCEPTION).release();
  }

  // Dynamic shape inference

  // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
  // tensors instead of binding them to the infer request directly.

  // Bind inputs
  for (const auto& input_info : onnx_to_ov_bindings_->network_inputs_) {
    // Set the input shape based on the input tensor from ort
    auto tensor = context.GetInput(input_info.onnx_index);
    auto ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto input_shape = ParameterShape(ort_shape);

    _infer_request->SetTensor(input_info.name,
                              input_info.type,
                              input_shape,
                              const_cast<void*>(tensor.GetTensorRawData()));
  }

  // pre-inference
  OVEP_RETURN_IF_ERROR(PreInfer());

  // Run Inference
  _infer_request->Infer();

  // Copy outputs
  for (const auto& output_info : onnx_to_ov_bindings_->network_outputs_) {
    auto ov_tensor = _infer_request->ov().get_tensor(output_info.name);
    auto output_shape = ParameterShape::ToOrtShape(ov_tensor.get_shape());
    auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);
    OVEP_RETURN_IF(ov_tensor.get_byte_size() != ort_tensor.GetTensorSizeInBytes(),
                   ort_api,
                   std::format("Output tensor size mismatch for {}", output_info.name).c_str());

    std::memcpy(ort_tensor.GetTensorMutableRawData(),
                ov_tensor.data(),
                ov_tensor.get_byte_size());
  }

  return nullptr;
}

OrtStatus* OvComputeInfoStateful::KVCacheRewind(const size_t& index) {
  auto& infer_request = _infer_request->ov();
  if (prefill_use_full_chat_history_) {
    // Clear the internal KVCache state. For NPU device, this operation is a no-op.
    infer_request.reset_state();

    // Resize the cached "input_ids" and "position_ids" to the specified index.
    if (cached_input_ids_.size() > index) {
      cached_input_ids_.resize(index);
    }

    if (cached_position_ids_.size() > index) {
      cached_position_ids_.resize(index);
    }
  } else {
    // If index == 0, we are clearing the entire KVCache, so simply reset the state.
    if (index == 0) {
      infer_request.reset_state();
      return nullptr;
    }
    // Retrieve KVCache states and trim them to the specified index.
    // The following logic is adapted from:
    // https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/1/src/cpp/src/utils.cpp#L329
    auto states = infer_request.query_state();
    for (auto& state : states) {
      ov::Tensor old_tensor = state.get_state();
      // Tensor shape: [batch_size, num_kv_heads, seq_len, head_size]
      auto shape = old_tensor.get_shape();

      if (shape[2] > index) {
        // Update the sequence length dimension to the specified index.
        shape[2] = index;

        ov::Coordinate new_shape_begin{0, 0, 0, 0};
        ov::Coordinate new_shape_end{shape};

        // Create a trimmed tensor with the updated shape.
        auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

        // Copy the trimmed tensor into a new tensor and update the state.
        ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
        trimmed_tensor.copy_to(new_tensor);

        state.set_state(new_tensor);
      }
    }
  }
  return nullptr;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
