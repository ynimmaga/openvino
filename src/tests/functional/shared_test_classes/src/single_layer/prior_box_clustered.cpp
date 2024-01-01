// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/prior_box_clustered.hpp"

namespace LayerTestsDefinitions {
std::string PriorBoxClusteredLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, imageShapes;
    std::string targetDevice;
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams,
        netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes,
        imageShapes,
        targetDevice) = obj.param;

    std::vector<float> widths, heights, variances;
    float step_width, step_height, step, offset;
    bool clip;
    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        step,
        offset,
        variances) = specParams;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="      << ov::test::utils::vec2str(inputShapes) << separator;
    result << "imageS="  << ov::test::utils::vec2str(imageShapes) << separator;
    result << "netPRC="  << netPrecision.name()   << separator;
    result << "inPRC="   << inPrc.name() << separator;
    result << "outPRC="  << outPrc.name() << separator;
    result << "inL="     << inLayout << separator;
    result << "outL="    << outLayout << separator;
    result << "widths="  << ov::test::utils::vec2str(widths)  << separator;
    result << "heights=" << ov::test::utils::vec2str(heights) << separator;
    result << "variances=";
    if (variances.empty())
        result << "()" << separator;
    else
        result << ov::test::utils::vec2str(variances) << separator;
    result << "stepWidth="  << step_width  << separator;
    result << "stepHeight=" << step_height << separator;
    result << "step="       << step << separator;
    result << "offset="     << offset      << separator;
    result << "clip="       << std::boolalpha << clip << separator;
    result << "trgDev="     << targetDevice;
    return result.str();
}

void PriorBoxClusteredLayerTest::SetUp() {
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams, netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes, imageShapes, targetDevice) = GetParam();

    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        step,
        offset,
        variances) = specParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes))};

    ov::op::v0::PriorBoxClustered::Attributes attributes;
    attributes.widths = widths;
    attributes.heights = heights;
    attributes.clip = clip;
    attributes.step_widths = step_width;
    attributes.step_heights = step_height;
    attributes.step = step;
    attributes.offset = offset;
    attributes.variances = variances;

    auto shape_of_1 = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
    auto shape_of_2 = std::make_shared<ov::op::v3::ShapeOf>(params[1]);
    auto priorBoxClustered = std::make_shared<ov::op::v0::PriorBoxClustered>(
        shape_of_1,
        shape_of_2,
        attributes);

    ngraph::ResultVector results{ std::make_shared<ov::op::v0::Result>(priorBoxClustered) };
    function = std::make_shared<ngraph::Function>(results, params, "PB_Clustered");
}
}  // namespace LayerTestsDefinitions
