// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/depth_to_space_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/builders.hpp"

#include <ngraph/function.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <ngraph/op/depth_to_space.hpp>

#include "ov_lpt_models/depth_to_space.hpp"

namespace LayerTestsDefinitions {

std::string DepthToSpaceTransformation::getTestCaseName(const testing::TestParamInfo<DepthToSpaceTransformationParams>& obj) {
    static std::map<ov::op::v0::DepthToSpace::DepthToSpaceMode, std::string> names = {
        {ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    std::string targetDevice;
    ov::op::v0::DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    std::tie(precision, inputShape, targetDevice, mode, blockSize) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShape, targetDevice, params) <<
        "_" << names[mode] << "_" << blockSize;
    return result.str();
}

void DepthToSpaceTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    ov::op::v0::DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    std::tie(precision, inputShape, targetDevice, mode, blockSize) = this->GetParam();

    if (inputShape.rank().is_dynamic() || inputShape.rank().get_length() != 4) {
        IE_THROW() << "not supported input shape size " << inputShape.rank();
    }

    function = ngraph::builder::subgraph::DepthToSpaceFunction::getOriginal(precision, inputShape, mode, blockSize);
}

TEST_P(DepthToSpaceTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
