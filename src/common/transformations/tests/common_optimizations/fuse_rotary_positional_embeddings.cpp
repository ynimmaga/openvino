// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

void name_node_and_output(const shared_ptr<Node>& op, const std::string& name) {
    op->set_friendly_name(name);
    op->output(0).set_names({name});
}

TEST_F(TransformationTestsF, FuseRPE) {
    {
        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_node_and_output(data, "source");
        auto sin = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_node_and_output(sin, "sin");
        auto cos = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_node_and_output(cos, "cos");
        auto axis = v0::Constant::create(element::i64, {}, {-1});
        auto split_lengths = v0::Constant::create(element::i64, {2}, {10, 10});
        auto split = make_shared<v1::VariadicSplit>(data, axis, split_lengths);

        auto minus_one = v0::Constant::create(element::f32, {}, {-1});
        auto negate = make_shared<v1::Multiply>(split->output(1), minus_one);

        auto concat = make_shared<v0::Concat>(OutputVector{negate, split->output(0)}, -1);

        auto mul_sin = make_shared<op::v1::Multiply>(concat, sin);
        auto mul_cos = make_shared<op::v1::Multiply>(data, cos);
        auto add = make_shared<op::v1::Add>(mul_cos, mul_sin);
        name_node_and_output(add, "rpe");

        model = std::make_shared<Model>(NodeVector{add}, ParameterVector{data, sin, cos});

        manager.register_pass<ov::pass::RPE_Fusion>();
    }
    {
        auto data = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_node_and_output(data, "source");
        auto sin = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_node_and_output(sin, "sin");
        auto cos = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        name_node_and_output(cos, "cos");
        auto rpe = make_shared<ov::op::internal::RPE>(data, sin, cos, -1);
        name_node_and_output(rpe, "rpe");
        model_ref = std::make_shared<Model>(NodeVector{rpe}, ParameterVector{data, sin, cos});
    }
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

TEST_F(TransformationTestsF, FuseRPESorcesAreMultiOutputed) {
    /* Transformation matcher searches for a single source as a beginning of the pattern:
                 VariadicSplit ...
     source ____/
                \
                 Multiply ...
    This test is designed to check that in case we feed VariadicSplit and Multiply from different outputs of the same
    node, the transformation won't happen since the source isn't the same
     */
    {
        auto data_ = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto sin = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());
        auto cos = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic());

        auto data = make_shared<v1::Split>(data_, v0::Constant::create(element::i64, {}, {-1}), 2);

        auto axis = v0::Constant::create(element::i64, {}, {-1});
        auto split_lengths = v0::Constant::create(element::i64, {2}, {10, 10});
        auto split = make_shared<v1::VariadicSplit>(data->output(0), axis, split_lengths);

        auto minus_one = v0::Constant::create(element::f32, {}, {-1});
        auto negate = make_shared<v1::Multiply>(split->output(1), minus_one);

        auto concat = make_shared<v0::Concat>(OutputVector{negate, split->output(0)}, -1);

        auto mul_sin = make_shared<op::v1::Multiply>(concat, sin);
        auto mul_cos = make_shared<op::v1::Multiply>(data->output(1), cos);
        auto add = make_shared<op::v1::Add>(mul_cos, mul_sin);

        model = std::make_shared<Model>(NodeVector{add}, ParameterVector{data_, sin, cos});

        manager.register_pass<ov::pass::RPE_Fusion>();
    }
}