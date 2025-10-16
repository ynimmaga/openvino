// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <string>
#include <map>
#include <optional>

#include "openvino/core/any.hpp"
#include "openvino/core/partial_shape.hpp"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace onnxruntime {
namespace openvino_ep_plugin {

using ConfigMap = std::map<std::string, ov::AnyMap>;
using ReshapeMap = std::map<std::string, ov::PartialShape>;

OrtStatus* ParsePluginLoadConfigOption(const OrtApi& ort_api, const Ort::Logger& logger, const std::string& config_str, ConfigMap& target_map);
OrtStatus* ParsePluginReshapeInputOption(const OrtApi& ort_api, const Ort::Logger& logger, const std::string& config_str, ReshapeMap& reshape_map);

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
