// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <string>
#include <cstdint>
#include <fstream>
#include <format>

#include "ov_provider.h"
#include "../common/ov_common_utils.h"

template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
      : objects_(object_ptr), count_(1), release_func_(release_func) {}

  DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
      : objects_(objects), count_(count), release_func_(release_func) {}

  ~DeferOrtRelease() {
    if (objects_ != nullptr && count_ > 0) {
      for (size_t i = 0; i < count_; ++i) {
        if (objects_[i] != nullptr) {
          release_func_(objects_[i]);
          objects_[i] = nullptr;
        }
      }
    }
  }

  OVEP_DISABLE_COPY_AND_MOVE(DeferOrtRelease);

  T** objects_ = nullptr;
  size_t count_ = 0;
  std::function<void(T*)> release_func_ = nullptr;
};

namespace onnxruntime {
namespace openvino_ep_plugin {

struct EpContextNode : ApiPtrs {
  size_t num_nodes{0};
  int64_t main_context{1};
  std::string ep_cache_context;
  int64_t embed_mode{1};
  std::string ep_sdk_version;
  std::string onnx_model_filename;
  std::string hardware_architecture;
  std::string partition_name;
  std::string source;
  std::string notes;
  int64_t max_size{0};

  enum class EpContextType {
    Native,
    OV_IR,
  };

  struct private_fields {
    EpContextType type{EpContextType::Native};
    std::string node_name{"OpenVINO_EP_Node"};
    std::filesystem::path epctx_dir{};
  } private_fields_;

  EpContextNode(ApiPtrs apis) : ApiPtrs(apis) {}

  OrtStatus* Init(const OrtNode* node, const std::filesystem::path& ep_context_path, const std::filesystem::path& model_path) {
    // Helper lambda to extract attribute values
    auto get_attr_int64 = [&](const char* name, int64_t default_val) -> int64_t {
      int64_t val = default_val;
      const OrtOpAttr* attr = nullptr;
      const OrtOpAttrType type = ORT_OP_ATTR_INT;

      OrtStatus* status = ort_api.Node_GetAttributeByName(node, name, &attr);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      size_t size_read = 0;
      status = ort_api.ReadOpAttr(attr, type, &val, sizeof(val), &size_read);
      if (status || size_read != sizeof(val)) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      return val;
    };

    auto get_attr_string = [&](const char* name) -> std::string {
      std::string val{};
      const OrtOpAttr* attr = nullptr;
      const OrtOpAttrType type = ORT_OP_ATTR_STRING;

      OrtStatus* status = ort_api.Node_GetAttributeByName(node, name, &attr);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }
      size_t required_size = 0;
      status = ort_api.ReadOpAttr(attr, type, nullptr, 0, &required_size);
      if (status) {
        // expect it to fail
        ort_api.ReleaseStatus(status);
      }

      val.resize(required_size);
      status = ort_api.ReadOpAttr(attr, type, val.data(), val.size(), &required_size);
      if (status) {
        ort_api.ReleaseStatus(status);
        return val;
      }

      return val;
    };

    main_context = get_attr_int64("main_context", 1);
    ep_cache_context = get_attr_string("ep_cache_context");
    embed_mode = get_attr_int64("embed_mode", 1);
    ep_sdk_version = get_attr_string("ep_sdk_version");
    onnx_model_filename = get_attr_string("onnx_model_filename");
    hardware_architecture = get_attr_string("hardware_architecture");
    partition_name = get_attr_string("partition_name");
    source = get_attr_string("source");
    notes = get_attr_string("notes");
    max_size = get_attr_int64("max_size", 0);

    private_fields_ = private_fields{};
    private_fields_.epctx_dir = ep_context_path.parent_path();

    if (embed_mode == 1) {
      if (openvino_ep::utils::IsXmlHeader(ep_cache_context)) {
        private_fields_.type = EpContextType::OV_IR;
      }
    } else {
      std::filesystem::path ep_ctx_path;
      if (ep_cache_context.ends_with(".xml")) {
        private_fields_.epctx_dir = model_path.parent_path();
        ep_ctx_path = private_fields_.epctx_dir / ep_cache_context;
      } else {
        ep_ctx_path = private_fields_.epctx_dir / ep_cache_context;
      }

      std::ifstream ep_ctx_file(ep_ctx_path, std::ios::binary);
      OVEP_RETURN_IF(ep_ctx_file.fail(), ort_api, std::format("Could not open EP context file {}", ep_ctx_path.string()).c_str());

      if (openvino_ep::utils::IsModelStreamXML(ep_ctx_file)) {
        private_fields_.type = EpContextType::OV_IR;
      }
    }

    return nullptr;
  }
  OrtStatus* CreateNode(const OnnxIOMapping& io_map, OrtNode*& node) {
    std::array<OrtOpAttr*, 7> attributes = {};
    DeferOrtRelease<OrtOpAttr> defer_release_attrs(attributes.data(), attributes.size(), ort_api.ReleaseOpAttr);

    OVEP_RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_cache_context", ep_cache_context.c_str(), ep_cache_context.size(), ORT_OP_ATTR_STRING, &attributes[0]));
    OVEP_RETURN_IF_ERROR(ort_api.CreateOpAttr("main_context", &main_context, 1, ORT_OP_ATTR_INT, &attributes[1]));
    OVEP_RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT, &attributes[2]));
    OVEP_RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_sdk_version", ep_sdk_version.c_str(), ep_sdk_version.size(), ORT_OP_ATTR_STRING, &attributes[3]));
    OVEP_RETURN_IF_ERROR(ort_api.CreateOpAttr("partition_name", partition_name.c_str(), partition_name.size(), ORT_OP_ATTR_STRING, &attributes[4]));
    OVEP_RETURN_IF_ERROR(ort_api.CreateOpAttr("source", source.c_str(), source.size(), ORT_OP_ATTR_STRING, &attributes[5]));
    OVEP_RETURN_IF_ERROR(ort_api.CreateOpAttr("onnx_model_filename", onnx_model_filename.c_str(), onnx_model_filename.size(), ORT_OP_ATTR_STRING, &attributes[6]));

    // Prepare input and output names
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    for (const auto& input : io_map.input_names) {
      input_names.push_back(input.c_str());
    }
    for (const auto& output : io_map.output_names) {
      output_names.push_back(output.c_str());
    }

    // Create the node
    OrtStatus* status = model_editor_api.CreateNode(
        "EPContext",
        "com.microsoft",
        private_fields_.node_name.c_str(),
        input_names.data(),
        input_names.size(),
        output_names.data(),
        output_names.size(),
        attributes.data(),
        attributes.size(),
        &node);

    return status;
  }
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
