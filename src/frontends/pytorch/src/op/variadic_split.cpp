// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/variadic_split.hpp"

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_split_with_sizes_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto sizes = context.get_input(1);

    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    } else {
        dim = context.get_input(2);
    }

    auto vsplit = context.mark_node(std::make_shared<v1::VariadicSplit>(context.get_input(0), dim, sizes));
    return {context.mark_node(make_list_construct(vsplit->outputs()))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
