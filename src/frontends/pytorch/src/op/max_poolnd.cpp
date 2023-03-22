// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_max_poolnd(NodeContext& context) {
    num_inputs_check(context, 4, 6);
    auto x = context.get_input(0);
    auto kernel = context.const_input<Shape>(1);
    auto strides = context.const_input<Strides>(2);
    auto pads = context.const_input<Shape>(3);  // pytorch supports only symmetric paddings
    Strides dilations;
    if (!context.input_is_none(4)) {
        dilations = context.const_input<Strides>(4);
    }
    RoundingType rounding_type;
    if (context.input_is_none(5)) {
        rounding_type = RoundingType::FLOOR;
    } else {
        rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL : RoundingType::FLOOR;
    }
    bool expand_input_shape = false;
    if (x.get_shape().size() == 3 && kernel.size() == 2) expand_input_shape = true;

    if (expand_input_shape) {
        std::vector<int> n_dim_vec = {1};
        auto n_dim_const = context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, n_dim_vec));
        auto input_shape = context.mark_node(std::make_shared<ov::op::v3::ShapeOf>(context.get_input(0), element::i32));
        auto new_input_shape = context.mark_node(std::make_shared<ov::op::v0::Concat>(OutputVector{n_dim_const, input_shape}, 0));

        x = context.mark_node(std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_input_shape, false));
    }

    auto max_pool = context.mark_node(std::make_shared<v8::MaxPool>(x, strides, dilations, pads, pads, kernel, rounding_type));

    if (expand_input_shape) {
        std::vector<int> out_shape = {max_pool->output(0).get_shape().begin()+1, max_pool->output(0).get_shape().end()};
        auto out_shape_node = context.mark_node(std::make_shared<ov::op::v0::Constant>(element::i32, Shape{3}, out_shape));
        max_pool = context.mark_node(std::make_shared<ov::op::v1::Reshape>(max_pool, out_shape_node, false));
    }

    return {max_pool};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
