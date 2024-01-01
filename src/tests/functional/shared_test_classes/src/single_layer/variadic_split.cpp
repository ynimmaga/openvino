// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/variadic_split.hpp"

namespace LayerTestsDefinitions {

    std::string VariadicSplitLayerTest::getTestCaseName(const testing::TestParamInfo<VariadicSplitParams>& obj) {
        int64_t axis;
        std::vector<size_t> numSplits;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InferenceEngine::SizeVector inputShapes;
        std::string targetDevice;
        std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        result << "numSplits=" << ov::test::utils::vec2str(numSplits) << "_";
        result << "axis=" << axis << "_";
        result << "IS";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

    void VariadicSplitLayerTest::SetUp() {
        int64_t axis;
        std::vector<size_t> inputShape, numSplits;
        InferenceEngine::Precision netPrecision;
        std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, axis);
        auto num_split = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{numSplits.size()}, numSplits);
        auto VariadicSplit = std::make_shared<ov::op::v1::VariadicSplit>(params[0], split_axis_op, num_split);

        ngraph::ResultVector results;
        for (int i = 0; i < numSplits.size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(VariadicSplit->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "VariadicSplit");
    }

}  // namespace LayerTestsDefinitions
