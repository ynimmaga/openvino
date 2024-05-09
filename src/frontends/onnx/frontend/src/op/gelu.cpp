// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/gelu.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/gelu.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector gelu(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();
    std::string approximate = node.get_attribute_value<std::string>("approximate", "none");

    FRONT_END_GENERAL_CHECK(inputs.size() == 1, "Wrong number of inputs, expected 1, found ", inputs.size());
    const auto input_type = inputs[0].get_element_type();
    FRONT_END_GENERAL_CHECK(input_type == ov::element::bf16 || input_type == ov::element::f16 ||
                                input_type == ov::element::f32 || input_type == ov::element::f64,
                            "Wrong input type, expected BFLOAT16, FLOAT16, FLOAT, DOUBLE, but found ",
                            input_type.get_type_name());
    FRONT_END_GENERAL_CHECK(approximate == "none" || approximate == "tanh",
                            "Unsupported approximate attribute: ",
                            approximate);

    return {std::make_shared<v7::Gelu>(
        inputs[0],
        approximate == "none" ? ov::op::GeluApproximationMode::ERF : ov::op::GeluApproximationMode::TANH)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
