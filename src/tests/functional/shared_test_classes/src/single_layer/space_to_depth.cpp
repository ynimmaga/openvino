// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/space_to_depth.hpp"

namespace LayerTestsDefinitions {

static inline std::string SpaceToDepthModeToString(const ov::op::v0::SpaceToDepth::SpaceToDepthMode& mode) {
    static std::map<ov::op::v0::SpaceToDepth::SpaceToDepthMode, std::string> names = {
        {ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    auto i = names.find(mode);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported SpaceToDepthMode");
}

std::string SpaceToDepthLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToDepthParamsTuple> &obj) {
    std::vector<size_t> inShape;
    ov::op::v0::SpaceToDepth::SpaceToDepthMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::string targetName;
    std::tie(inShape, inputPrecision, mode, blockSize, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inShape) << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "M=" << SpaceToDepthModeToString(mode) << "_";
    result << "BS=" << blockSize << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void SpaceToDepthLayerTest::SetUp() {
    std::vector<size_t> inShape;
    ov::op::v0::SpaceToDepth::SpaceToDepthMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::tie(inShape, inputPrecision, mode, blockSize, targetDevice) = this->GetParam();
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(inPrc, ov::Shape(inShape))};
    auto s2d = std::make_shared<ov::op::v0::SpaceToDepth>(params[0], mode, blockSize);
    ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(s2d)};
    function = std::make_shared<ngraph::Function>(results, params, "SpaceToDepth");
}
}  // namespace LayerTestsDefinitions
