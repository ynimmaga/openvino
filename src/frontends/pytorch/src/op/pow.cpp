// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pow(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto inplace = context.get_op_type() == "aten::pow_";
    if (inplace) {
        rhs = std::make_shared<ov::op::v1::ConvertLike>(rhs, lhs);
    } else {
        align_eltwise_input_types(context, lhs, rhs, true);
    }
    auto res = context.mark_node(std::make_shared<ov::op::v1::Power>(lhs, rhs));
    if (inplace) {
        context.mutate_input(0, res);
    }
    return {res};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
