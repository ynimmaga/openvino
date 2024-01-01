// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/einsum.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

std::string EinsumLayerTest::getTestCaseName(const testing::TestParamInfo<EinsumLayerTestParamsSet>& obj) {
    InferenceEngine::Precision precision;
    EinsumEquationWithInput equationWithInput;
    std::string targetDevice;
    std::tie(precision, equationWithInput, targetDevice) = obj.param;
    std::string equation;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(equation, inputShapes) = equationWithInput;

    std::ostringstream result;
    result << "PRC=" << precision.name() << "_";
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "Eq=" << equation << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void EinsumLayerTest::SetUp() {
    InferenceEngine::Precision precision;
    EinsumEquationWithInput equationWithInput;
    std::tie(precision, equationWithInput, targetDevice) = this->GetParam();
    std::string equation;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(equation, inputShapes) = equationWithInput;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);
    ov::ParameterVector params;
    ov::OutputVector paramsOuts;
    for (auto&& shape : inputShapes) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape));
        params.push_back(param);
        paramsOuts.push_back(param);
    }

    const auto einsum = std::make_shared<ov::op::v7::Einsum>(paramsOuts, equation);
    const ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(einsum)};
    function = std::make_shared<ngraph::Function>(results, params, "einsum");
}

}  // namespace LayerTestsDefinitions
