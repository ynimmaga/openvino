// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_binary_op(const NodeContext& node,
                                 const std::function<Output<Node>(Output<Node>&, Output<Node>&)>& create_binary_op) {
    default_op_checks(node, 2, {});
    auto lhs = node.get_input(0);
    auto rhs = node.get_input(1);
    auto result = create_binary_op(lhs, rhs);
    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}

OutputVector translate_floor_div_op(const NodeContext& node) {
    auto floordiv_fn = [](const Output<Node>& x, const Output<Node>& y) -> shared_ptr<Node> {
        auto out_type = x.get_element_type();
        if (out_type.is_integral()) {
            auto float_x = make_shared<v0::Convert>(x, element::f32);
            auto float_y = make_shared<v0::Convert>(y, element::f32);
            return make_shared<v0::Convert>(make_shared<v0::Floor>(make_shared<v1::Divide>(float_x, float_y)),
                                            out_type);
        } else {
            return make_shared<v0::Floor>(make_shared<v1::Divide>(x, y));
        }
    };
    return translate_binary_op(node, floordiv_fn);
}

template <typename T>
OutputVector translate_binary_op(const NodeContext& node) {
    return translate_binary_op(node, [](Output<Node>& ng_lhs, Output<Node>& ng_rhs) {
        return make_shared<T>(ng_lhs, ng_rhs);
    });
}

OutputVector translate_mul_op(const NodeContext& node) {
    default_op_checks(node, 2, {}, true);
    auto lhs = node.get_input(0);
    auto rhs = node.get_input(1);
    auto result = make_shared<v1::Multiply>(lhs, rhs);

    auto complex_type_mark_lhs = as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto complex_type_mark_rhs = as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());
    if (complex_type_mark_lhs || complex_type_mark_rhs) {
        FRONT_END_GENERAL_CHECK(complex_type_mark_lhs != nullptr && complex_type_mark_rhs != nullptr,
                                "Mul gox complex and non-complex inputs. Inputs should be of same type.");
        lhs = complex_type_mark_lhs->input_value(0);
        rhs = complex_type_mark_rhs->input_value(0);

        element::Type complex_part_type_lhs = complex_type_mark_lhs->get_complex_part_type();
        element::Type complex_part_type_rhs = complex_type_mark_rhs->get_complex_part_type();
        FRONT_END_GENERAL_CHECK(complex_part_type_lhs == complex_part_type_rhs,
                                "Mul got complex inputs of different types. Inputs should be of same type.");

        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);

        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

        auto lhs_real = make_shared<v8::Gather>(lhs, gather_index_real, minus_one)->output(0);
        auto lhs_imag = make_shared<v8::Gather>(lhs, gather_index_imag, minus_one)->output(0);
        auto rhs_real = make_shared<v8::Gather>(rhs, gather_index_real, minus_one)->output(0);
        auto rhs_imag = make_shared<v8::Gather>(rhs, gather_index_imag, minus_one)->output(0);

        // result_real = lhs_real * rhs_real - lhs_imag * rhs_imag
        auto result_real = make_shared<v1::Subtract>(make_shared<v1::Multiply>(lhs_real, rhs_real),
                                                     make_shared<v1::Multiply>(lhs_imag, rhs_imag));

        // result_imag = lhs_real * rhs_imag + lhs_imag * rhs_real
        auto result_imag = make_shared<v1::Add>(make_shared<v1::Multiply>(lhs_real, rhs_imag),
                                                make_shared<v1::Multiply>(lhs_imag, rhs_real));

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(result_real, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(result_imag, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        set_node_name(node.get_name(), concat_result);

        auto complex_result = make_shared<ComplexTypeMark>(concat_result->output(0), complex_part_type_lhs);
        return {complex_result};
    }

    set_node_name(node.get_name(), result);
    return {result};
}
template OutputVector translate_binary_op<v1::Add>(const NodeContext& node);
template OutputVector translate_binary_op<v13::BitwiseAnd>(const NodeContext& node);
template OutputVector translate_binary_op<v13::BitwiseOr>(const NodeContext& node);
template OutputVector translate_binary_op<v13::BitwiseXor>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Equal>(const NodeContext& node);
template OutputVector translate_binary_op<v1::FloorMod>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Greater>(const NodeContext& node);
template OutputVector translate_binary_op<v1::GreaterEqual>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Less>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LessEqual>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LogicalAnd>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LogicalOr>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LogicalXor>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Maximum>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Minimum>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Multiply>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Mod>(const NodeContext& node);
template OutputVector translate_binary_op<v1::NotEqual>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Power>(const NodeContext& node);
template OutputVector translate_binary_op<v0::PRelu>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Divide>(const NodeContext& node);
template OutputVector translate_binary_op<v0::SquaredDifference>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Subtract>(const NodeContext& node);

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
