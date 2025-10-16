// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "plugin/transformations/ov_weights_as_input.h"
#include "plugin/ov_utils.h"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

using ov_const_set_t = std::set<std::shared_ptr<ov::op::v0::Constant>, std::owner_less<std::shared_ptr<ov::op::v0::Constant>>>;

static std::unordered_set<std::shared_ptr<ov::op::v0::Constant>> FindDequantScaleZPConstants(std::shared_ptr<ov::Model>& model) {
  // Create a pattern to identify dequantize linear.
  using namespace ov::pass::pattern;

  // Zero point path: Convert -> (optional Reshape) -> Subtract
  auto zero_point = any_input();
  auto zp_convert = wrap_type<ov::op::v0::Convert>({zero_point});
  auto zp_optional_reshape = optional<ov::op::v1::Reshape>({zp_convert, any_input()});
  auto subtract = wrap_type<ov::op::v1::Subtract>({any_input(), zp_optional_reshape});

  // Scale path: (optional Reshape) -> Multiply
  auto scale = any_input();
  auto scale_optional_reshape = optional<ov::op::v1::Reshape>({scale, any_input()});
  auto multiply = wrap_type<ov::op::v1::Multiply>({subtract, scale_optional_reshape});

  auto matcher = std::make_shared<ov::pass::pattern::Matcher>(multiply, "dequantize_pattern");

  std::unordered_set<std::shared_ptr<ov::op::v0::Constant>> dequant_scale_zp_consts;
  for (auto&& node : model->get_ordered_ops()) {
    // Only need to run the pattern on Multiply nodes
    if (!std::dynamic_pointer_cast<ov::op::v1::Multiply>(node)) continue;
    if (!matcher->match(node->output(0))) continue;

    auto& pattern_map = matcher->get_pattern_value_map();

    // Check if scale is a constant
    auto scale_node = pattern_map.at(scale).get_node_shared_ptr();
    if (auto scale_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(scale_node)) {
      dequant_scale_zp_consts.insert(scale_constant);
    }

    // Check if zero_point source is a constant
    auto zero_point_source = pattern_map.at(zero_point).get_node_shared_ptr();
    if (auto zp_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(zero_point_source)) {
      dequant_scale_zp_consts.insert(zp_constant);
    }
  }
  return dequant_scale_zp_consts;
}

std::shared_ptr<ov::Model> ConvertConstantsToInputs(std::shared_ptr<ov::Model> model, SharedContext::SharedWeights& shared_weights) {
  // Zero points and scales of dequantize operations will not be converted to inputs
  auto dequant_scale_zp_consts = FindDequantScaleZPConstants(model);

  // Convert each constant to a parameter that are NOT dequantization parameters
  std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_parameters;
  for (auto&& node : model->get_ordered_ops()) {
    auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
    // Only convert constants that are known shared weights.
    if (!constant || !shared_weights.IsSharedWeight(constant->get_friendly_name())) continue;
    // Only convert constant if it's not a dequantization scale or zero_point
    if (dequant_scale_zp_consts.contains(constant)) continue;

    // Create parameter with same shape and type as constant
    auto param_name = constant->get_friendly_name();
    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        constant->get_element_type(),
        constant->get_shape());
    parameter->set_friendly_name(param_name);

    // Replace all uses of constant with the new parameter
    constant->output(0).replace(parameter->output(0));

    new_parameters.push_back(parameter);
  }

  // Get existing parameters and add new ones
  auto original_params = model->get_parameters();
  original_params.insert(original_params.end(), new_parameters.begin(), new_parameters.end());

  // Create new model with additional parameters
  auto new_model = std::make_shared<ov::Model>(
      model->get_results(),
      original_params,
      model->get_friendly_name() + "_weights_as_inputs");

  return new_model;
}

OrtStatus* weights_as_inputs::TransformSharedWeightsToInputs(std::shared_ptr<ov::Model>& model, SharedContext::SharedWeights& shared_weights) {
  try {
    model = ConvertConstantsToInputs(model, shared_weights);
    return nullptr;  // Success
  } catch (const std::exception& e) {
    return Ort::Status(OVEP_ERROR_STR("Failed to transform model weights: ", e.what()).c_str(),
                       ORT_RUNTIME_EXCEPTION);
  } catch (...) {
    return Ort::Status(OVEP_ERROR_STR("Failed to transform model weights: unknown error").c_str(), ORT_RUNTIME_EXCEPTION);
  }
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
