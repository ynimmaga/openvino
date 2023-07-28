// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_mul_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto dtype0 = context.get_input_type(0);
    auto dtype1 = context.get_input_type(1);
    if (dtype0.is<type::List>() && dtype1.is<type::List>()) {
        // aten::add.t(t[] a, t[] b) -> t[]
        // Case when two lists gets concatenated
        FRONT_END_OP_CONVERSION_CHECK(false, "aten::add is used for concatenation of lists, not possible to convert");
    }
    lhs = context.mark_node(std::make_shared<v0::Convert>(lhs, element::f32));
    rhs = context.mark_node(std::make_shared<v0::Convert>(rhs, element::f32));
    align_eltwise_input_types(context, lhs, rhs, true);
    /*if (!context.input_is_none(2)) {
        auto converted_alpha = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(context.get_input(2), rhs));
        rhs = context.mark_node(std::make_shared<ov::op::v1::Multiply>(converted_alpha, rhs));
    }*/
    return {context.mark_node(std::make_shared<ov::op::v1::Multiply>(lhs, rhs))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
