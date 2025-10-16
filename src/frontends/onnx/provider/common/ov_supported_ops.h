// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <string_view>
#include <set>

#include "ov_version.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

struct SupportedOps {
  bool
  IsOpSupported(std::string_view op_type, std::string_view domain) const {
    return ops_.find(std::make_pair(op_type, domain)) != ops_.end();
  }

  bool IsEpContextNode(std::string_view op_type, std::string_view domain) const {
    return op_type == "EPContext" && domain == "com.microsoft";
  }

  static SupportedOps& Get() {
    static SupportedOps instance(OpenVINOVersion::GetBuiltVersion());
    return instance;
  }

 private:
  SupportedOps() = default;
  SupportedOps(const OpenVINOVersion& ov_version);
  OpenVINOVersion version_;
  std::set<std::pair<std::string_view, std::string_view>> ops_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
