// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <sstream>
#include <set>
#include <regex>
#include <algorithm>
#include <format>
#include "nlohmann/json.hpp"
#include "ov_plugin_utils.h"
#include "../common/ov_common_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

OrtStatus* ParsePluginLoadConfigOption(const OrtApi& ort_api, const Ort::Logger& logger, const std::string& config_str, ConfigMap& target_map) {
  if (config_str.empty()) {
    ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_WARNING, "Empty OV Config Map passed. Skipping load_config option parsing.");
    target_map = {};
    return nullptr;
  }

  std::stringstream input_str_stream(config_str);
  try {
    nlohmann::json json_config = nlohmann::json::parse(input_str_stream);

    if (!json_config.is_object()) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid JSON structure: Expected an object at the root.");
    }

    for (const auto& [key, value] : json_config.items()) {
      ov::AnyMap inner_map;

      // Ensure that the value for each device is an object (PROPERTY -> VALUE)
      if (!value.is_object()) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                    "Invalid JSON structure: Expected an object for device properties.");
      }

      for (auto& [inner_key, inner_value] : value.items()) {
        if (inner_value.is_string()) {
          inner_map[inner_key] = ov::Any(inner_value.get<std::string>());
        } else if (inner_value.is_number_integer()) {
          inner_map[inner_key] = ov::Any(inner_value.get<int64_t>());
        } else if (inner_value.is_number_float()) {
          inner_map[inner_key] = ov::Any(inner_value.get<double>());
        } else if (inner_value.is_boolean()) {
          inner_map[inner_key] = ov::Any(inner_value.get<bool>());
        } else {
          ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_WARNING,
                       "Unsupported JSON value type for key: %s. Skipping key.", inner_key.c_str());
        }
      }
      target_map[key] = std::move(inner_map);
    }
  } catch (const nlohmann::json::parse_error& e) {
    // Handle syntax errors in JSON
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("JSON parsing error: {}", e.what()).c_str());
  } catch (const nlohmann::json::type_error& e) {
    // Handle invalid type accesses
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("JSON type error: {}", e.what()).c_str());
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("Error parsing load_config Map: {}", e.what()).c_str());
  }

  return nullptr;
}

OrtStatus* ParsePluginReshapeInputOption(const OrtApi& ort_api, const Ort::Logger& /*logger*/, const std::string& config_str, ReshapeMap& reshape_map) {
  // Return empty map for empty input
  if (config_str.empty()) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("reshape_input parameter: Invalid input shape definition \"{}\"", config_str).c_str());
  }

  try {
    // Regular expressions for parsing
    const std::regex tensor_pattern(R"(([^\[\],]+)\s*\[(.*?)\])");  // e.g. "input_1[1..5, 2, 3..4],data[1,2,3]"
    const std::regex dimension_pattern(R"(\s*([^,\s]+)\s*)");

    // Find all tensor shape definitions using regex
    auto tensor_begin = std::sregex_iterator(
        config_str.begin(),
        config_str.end(),
        tensor_pattern);
    auto tensor_end = std::sregex_iterator();

    // If no matches found, throw error
    if (tensor_begin == tensor_end) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                  std::format("reshape_input parameter: Invalid input shape definition format: {}", config_str).c_str());
    }

    // Process each tensor definition e.g. "input_1[1..5, 2, 3..4],data[1,2,3]"
    for (std::sregex_iterator i = std::move(tensor_begin); i != tensor_end; ++i) {
      std::smatch tensor_match = *i;

      // Extract tensor name and trim whitespace
      std::string tensor_name = tensor_match[1].str();  // Group 1: tensor name e.g. "input_1"
      tensor_name = onnxruntime::openvino_ep::utils::TrimWhitespace(tensor_name);

      if (tensor_name.empty()) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "reshape_input parameter: Empty tensor name provided in reshape_input parameter");
      }

      // Extract dimensions string
      std::string dimensions_str = tensor_match[2].str();  // Group 2: dimensions string [e.g. "1..5, 2, 3..4"]
      std::vector<ov::Dimension> dimensions;

      // Find all dimension e.g. "1..5", "2", "3..4" using regex
      auto dim_begin = std::sregex_iterator(
          dimensions_str.begin(),
          dimensions_str.end(),
          dimension_pattern);
      auto dim_end = std::sregex_iterator();

      if (dim_begin == dim_end) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, std::format("reshape_input parameter: Empty tensor dimensions for tensor: {}", tensor_name).c_str());
      }

      // Process each dimension
      for (std::sregex_iterator j = std::move(dim_begin); j != dim_end; ++j) {
        std::smatch dim_match = *j;
        std::string dim_value = dim_match[1].str();

        // Check if dimension is a range
        size_t range_separator_pos = dim_value.find("..");
        if (range_separator_pos != std::string::npos) {
          // Parse range
          auto range_pair = onnxruntime::openvino_ep::utils::ParseDimensionRange(dim_value, tensor_name);
          dimensions.push_back(ov::Dimension(range_pair.first, range_pair.second));
        } else {
          // Parse single value
          bool is_valid_integer = !dim_value.empty() &&
                                  std::all_of(dim_value.begin(), dim_value.end(), [](char c) {
                                    return std::isdigit(static_cast<unsigned char>(c));
                                  });

          if (!is_valid_integer) {
            return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        std::format("reshape_input parameter: Invalid dimension value: '{}' for tensor: {}", dim_value, tensor_name).c_str());
          }

          dimensions.push_back(std::stoi(dim_value));
        }
      }

      // Store parsed shape in result map
      reshape_map[tensor_name] = ov::PartialShape(std::move(dimensions));
    }
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                std::format("reshape_input parameter: Error parsing reshape_input: {}", e.what()).c_str());
  }

  return nullptr;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
