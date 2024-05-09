// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using RemoveConvertCPUTestParams = std::tuple<ElementType, InputShape>;

class RemoveUselessBF16ConvertCPUTest : public testing::WithParamInterface<RemoveConvertCPUTestParams>,
                                        virtual public SubgraphBaseTest,
                                        public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RemoveConvertCPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        if (inType == ElementType::bf16) {
            configuration.insert({ov::hint::inference_precision(ov::element::bf16)});
        }
        std::tie(inFmts, outFmts, priority, selectedType) =
            CPUSpecificParams{{}, {}, {}, makeSelectedTypeStr("ref", inType)};
        init_input_shapes({inputShape});
        auto input_params = std::make_shared<ov::op::v0::Parameter>(inType, inputShape.first);
        auto convert = std::make_shared<ov::op::v0::Convert>(input_params, element::f32);
        auto begin = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, 0});
        auto end = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 16, 0});
        auto stride = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
        auto slice = std::make_shared<ov::op::v1::StridedSlice>(convert,
                                                                begin,
                                                                end,
                                                                stride,
                                                                std::vector<int64_t>{0, 0, 0, 0},
                                                                std::vector<int64_t>{1, 1, 0, 1},
                                                                std::vector<int64_t>{},
                                                                std::vector<int64_t>{},
                                                                std::vector<int64_t>{});
        auto convert2 = std::make_shared<ov::op::v0::Convert>(slice, inType);
        function = std::make_shared<ov::Model>(convert2, ov::ParameterVector{input_params}, "remove_convert");
    };
};

class RemoveUselessConvertCPUTest : public testing::WithParamInterface<RemoveConvertCPUTestParams>,
                                    virtual public SubgraphBaseTest,
                                    public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RemoveConvertCPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes({inputShape});
        auto input_params = std::make_shared<ov::op::v0::Parameter>(inType, inputShape.first);

        // Such complicated graph is necessary to cover the case when Convert has several children and connected to non
        // zero output
        const auto split_axis = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{}, std::vector<int64_t>{1});
        const auto split_lengths = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1});
        const auto split = std::make_shared<ov::op::v1::VariadicSplit>(input_params, split_axis, split_lengths);
        auto convert = std::make_shared<ov::op::v0::Convert>(split->output(1), inType);
        auto relu = ov::test::utils::make_activation(convert, inType, ov::test::utils::ActivationTypes::Relu);

        ov::ResultVector results{
            std::make_shared<ov::op::v0::Result>(split->output(0)),
            std::make_shared<ov::op::v0::Result>(convert),
            std::make_shared<ov::op::v0::Result>(relu),
        };

        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input_params}, "remove_convert");
    };
};

TEST_P(RemoveUselessBF16ConvertCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithTypes(compiledModel, {"Convert", "Subgraph"}, 0);
    CheckPluginRelatedResults(compiledModel, "StridedSlice");
}

TEST_P(RemoveUselessConvertCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
}

namespace {
const std::vector<InputShape> inputShapes = {
    // dynamic batch
    {{-1, 4, 32, 64}, {{1, 4, 32, 64}, {2, 4, 32, 64}, {3, 4, 32, 64}}},
    {{-1, -1, -1, -1}, {{1, 4, 32, 64}, {2, 4, 32, 64}, {3, 4, 32, 64}}},
    // static shape
    {{1, 4, 32, 64}, {{1, 4, 32, 64}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RemoveConvert,
                         RemoveUselessBF16ConvertCPUTest,
                         ::testing::Combine(::testing::Values(ElementType::bf16), ::testing::ValuesIn(inputShapes)),
                         RemoveUselessBF16ConvertCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RemoveConvert,
                         RemoveUselessConvertCPUTest,
                         ::testing::Combine(::testing::Values(ElementType::f32), ::testing::Values(inputShapes[0])),
                         RemoveUselessConvertCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
