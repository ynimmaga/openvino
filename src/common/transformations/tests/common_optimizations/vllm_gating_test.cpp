// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Covers the "vllm_model" rt_info gate around NormalizeVLLMRoPE: other
// CommonOptimizations callers must not see it fire.
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"

namespace {

// cos/sin half-sized to match each split output (see normalize_vllm_rope_test.cpp).
const ov::Shape full_shape{1, 4, 8};
const ov::Shape half_shape{1, 4, 4};

std::shared_ptr<ov::Model> build_vllm_rope_model() {
    auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, full_shape);
    auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);
    auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    auto split = std::make_shared<ov::op::v1::Split>(x, axis, 2);

    auto mul_a = std::make_shared<ov::op::v1::Multiply>(split->output(0), cos);
    auto mul_b = std::make_shared<ov::op::v1::Multiply>(split->output(1), sin);
    auto neg_one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto neg = std::make_shared<ov::op::v1::Multiply>(mul_b, neg_one);
    auto sub_branch = std::make_shared<ov::op::v1::Add>(mul_a, neg);

    auto mul_c = std::make_shared<ov::op::v1::Multiply>(split->output(1), cos);
    auto mul_d = std::make_shared<ov::op::v1::Multiply>(split->output(0), sin);
    auto add_branch = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub_branch, add_branch}, -1);
    return std::make_shared<ov::Model>(concat, ov::ParameterVector{x, cos, sin});
}

// The rotate-half form NormalizeVLLMRoPE emits.
std::shared_ptr<ov::Model> build_normalized_rope_model() {
    auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, full_shape);
    auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);
    auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);
    auto cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos, cos}, -1);
    auto sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin, sin}, -1);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {4, 4});
    auto vsplit = std::make_shared<ov::op::v1::VariadicSplit>(x, axis, lengths);

    auto neg_one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto x2_neg = std::make_shared<ov::op::v1::Multiply>(vsplit->output(1), neg_one);
    auto x_rot = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, vsplit->output(0)}, -1);
    auto x_cos = std::make_shared<ov::op::v1::Multiply>(x, cos_full);
    auto xrot_sin = std::make_shared<ov::op::v1::Multiply>(x_rot, sin_full);
    auto out = std::make_shared<ov::op::v1::Add>(x_cos, xrot_sin);
    return std::make_shared<ov::Model>(out, ov::ParameterVector{x, cos, sin});
}

}  // namespace

// Without the flag the model must come out of CommonOptimizations unchanged
// (model_ref defaults to a copy of the input model).
TEST_F(TransformationTestsF, VLLMGatingRoPEPassSkippedWithoutVLLMModelFlag) {
    model = build_vllm_rope_model();
    manager.register_pass<ov::pass::CommonOptimizations>();
}

TEST_F(TransformationTestsF, VLLMGatingRoPEPassAppliedWithVLLMModelFlag) {
    // NormalizeVLLMRoPE's copy_runtime_info() omits some new nodes (see normalize_vllm_rope_test.cpp).
    disable_rt_info_check();
    model = build_vllm_rope_model();
    model->set_rt_info(true, "vllm_model");
    manager.register_pass<ov::pass::CommonOptimizations>();
    model_ref = build_normalized_rope_model();
}
