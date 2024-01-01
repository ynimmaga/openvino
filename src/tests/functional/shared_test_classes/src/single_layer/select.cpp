// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/select.hpp"

namespace LayerTestsDefinitions {
    enum { CONDITION, THEN, ELSE, numOfInputs };

    std::string SelectLayerTest::getTestCaseName(const testing::TestParamInfo<selectTestParams> &obj) {
        std::vector<std::vector<size_t>> dataShapes(3);
        InferenceEngine::Precision dataType;
        ov::op::AutoBroadcastSpec broadcast;
        std::string targetDevice;
        std::tie(dataShapes, dataType, broadcast, targetDevice) = obj.param;
        std::ostringstream result;
        result << "COND=BOOL_" << ov::test::utils::vec2str(dataShapes[CONDITION]);
        result << "_THEN=" << dataType.name() << "_" << ov::test::utils::vec2str(dataShapes[THEN]);
        result << "_ELSE=" << dataType.name() << "_" << ov::test::utils::vec2str(dataShapes[ELSE]);
        result << "_" << broadcast.m_type;
        result << "_targetDevice=" << targetDevice;
        return result.str();
    }

    void SelectLayerTest::SetUp() {
        std::vector<std::vector<size_t>> inputShapes(numOfInputs);
        InferenceEngine::Precision inputPrecision;
        ov::op::AutoBroadcastSpec broadcast;
        std::tie(inputShapes, inputPrecision, broadcast, targetDevice) = this->GetParam();

        ngraph::ParameterVector paramNodesVector;
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(ngraph::element::Type_t::boolean, ngraph::Shape(inputShapes[CONDITION]));
        paramNodesVector.push_back(paramNode);
        auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
        for (size_t i = 1; i < inputShapes.size(); i++) {
            paramNode = std::make_shared<ov::op::v0::Parameter>(inType, ngraph::Shape(inputShapes[i]));
            paramNodesVector.push_back(paramNode);
        }
        auto select = std::make_shared<ov::op::v1::Select>(paramNodesVector[0], paramNodesVector[1], paramNodesVector[2], broadcast);
        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(select)};
        function = std::make_shared<ngraph::Function>(results, paramNodesVector, "select");
    }
}  // namespace LayerTestsDefinitions
