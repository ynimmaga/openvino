// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
// Subgraph:
/*
 *         params[0]   params[1]
 *             |          |
 * constant  shapeOf     /
 *      \      |        /
 *       broadcast     /
 *            \       /
 *             \     /
 *             reshape
 *                |
 *              result
 *
 *  This test is designed for correctness of reshape's in-place implementation.
 *
 *  Due to non-const target shape parameter (params[1]), reshape node
 *  is non-constant node even though the input tensor is constant node.
 *
 *  some logic protecting constant data from being corrupted by
 *  the in-place consumer may breaks the in-place assumption, and reshape
 *  should be able to handle this case correctly.
 */

class InPlaceReshapeFromConstantCheck : public SubgraphBaseTest {
protected:
    void SetUp() override {
        const auto rtPrc = ov::element::f32;
        const ov::Shape inpShape = {21660, 4};
        const ov::Shape secShape = {4};
        targetStaticShapes = {{inpShape, secShape}};
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(rtPrc, inpShape),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::i32, secShape)};
        auto shape = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
        auto c = ov::test::utils::deprecated::make_constant<float>(rtPrc, {}, {1.0f});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(c, shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(broadcast, params[1], false);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape->output(0))};
        function = std::make_shared<ov::Model>(results, params, "reshape_check");
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;
            if (i == 1) {
                tensor = ov::runtime::Tensor{ov::element::i32, targetInputStaticShapes[i]};
                auto inputData = tensor.data<ov::element_type_traits<ov::element::i32>::value_type>();
                const std::vector<unsigned> data = {38, 38, 15, 4};
                for (size_t j = 0lu; j < data.size(); ++j) {
                    inputData[j] = data[j];
                }
            } else {
                if (funcInput.get_element_type().is_real()) {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
                } else {
                    tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(InPlaceReshapeFromConstantCheck, smoke_CPU_InPlaceReshapeFromConstantCheck) {
    run();
}
}  // namespace test
}  // namespace ov
