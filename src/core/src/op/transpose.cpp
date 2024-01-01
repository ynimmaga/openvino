// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/reference/transpose.hpp"
#include "transpose_shape_inference.hpp"
#include "validation_util.hpp"

namespace ov {
namespace op {
namespace v1 {

Transpose::Transpose(const Output<Node>& arg, const Output<Node>& input_order) : Op({arg, input_order}) {
    constructor_validate_and_infer_types();
}

void Transpose::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Transpose_validate_and_infer_types);
    const auto& input_order_et = get_input_element_type(ORDER);
    NODE_VALIDATION_CHECK(this,
                          input_order_et.is_dynamic() || input_order_et.is_integral_number(),
                          "Input order must have an integral number element type.");

    set_input_is_relevant_to_shape(ORDER);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(ARG, get_input_element_type(ARG), output_shapes[ARG_T]);
}

std::shared_ptr<Node> Transpose::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Transpose_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Transpose>(new_args[ARG], new_args[ORDER]);
}

bool Transpose::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Transpose_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto& order = inputs[ORDER];
    if (order.get_element_type().is_integral()) {
        const auto& arg = inputs[ARG];
        auto axes_order = ov::get_tensor_data_as<int64_t>(order);
        const auto out_shape = calc_output_shape(this, arg.get_shape(), axes_order);

        auto& out = outputs[ARG_T];
        out.set_shape(out_shape);
        reference::transpose(static_cast<const char*>(arg.data()),
                             static_cast<char*>(out.data()),
                             arg.get_shape(),
                             arg.get_element_type().size(),
                             axes_order,
                             out_shape);
        return true;
    } else {
        return false;
    }
}

bool Transpose::has_evaluate() const {
    OV_OP_SCOPE(v1_Transpose_has_evaluate);
    return get_input_element_type(ORDER).is_integral_number();
}

bool Transpose::evaluate_lower(ov::TensorVector& output_values) const {
    return get_input_tensor(ORDER).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool Transpose::evaluate_upper(ov::TensorVector& output_values) const {
    return get_input_tensor(ORDER).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool Transpose::evaluate_label(TensorLabelVector& output_labels) const {
    return get_input_tensor(ORDER).has_and_set_bound() && ov::util::default_label_evaluator(this, output_labels);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
