// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/conv_concat.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/node_builders/group_convolution_backprop_data.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string ConvConcatSubgraphTest::getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj) {
    std::ostringstream result;
    nodeType type;
    commonConvParams convParams;
    CPUSpecificParams cpuParams;
    ov::Shape inputShapes;
    int axis;
    std::tie(type, convParams, cpuParams, inputShapes, axis) = obj.param;

    result << "Type=" << nodeType2str(type) << "_";

    std::vector<size_t> kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels, numOfGroups;
    ov::op::PadType paddingType;
    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType, numOfGroups) = convParams;

    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernelSize) << "_";
    result << "S" << ov::test::utils::vec2str(strides) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << numOutChannels << "_";
    result << "G=" << numOfGroups << "_";
    result << "AP=" << paddingType << "_";

    result << CPUTestsBase::getTestCaseName(cpuParams);

    result << "_axis=" << axis;

    return result.str();
}

void ConvConcatSubgraphTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    nodeType type;
    commonConvParams convParams;
    CPUSpecificParams cpuParams;
    ov::Shape inputShapes;
    int axis;

    std::tie(type, convParams, cpuParams, inputShapes, axis) = this->GetParam();
    pluginTypeNode = nodeType2PluginType(type);
    std::vector<size_t> kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels, numOfGroups;
    ov::op::PadType paddingType;

    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType, numOfGroups) = convParams;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    selectedType += "_f32";

    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShapes),
                                    std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShapes)};

    std::vector<std::shared_ptr<ov::Node>> convolutionNodes(2);
    switch (type) {
        case nodeType::convolution : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_convolution(inputParams[conv],
                                                                           ov::element::f32,
                                                                           kernelSize,
                                                                           strides,
                                                                           padBegin,
                                                                           padEnd,
                                                                           dilation,
                                                                           paddingType,
                                                                           numOutChannels);
            }
            break;
        }
        case nodeType::convolutionBackpropData : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_convolution_backprop_data(inputParams[conv],
                                                                                         ov::element::f32,
                                                                                         kernelSize,
                                                                                         strides,
                                                                                         padBegin,
                                                                                         padEnd,
                                                                                         dilation,
                                                                                         paddingType,
                                                                                         numOutChannels);
            }
            break;
        }
        case nodeType::groupConvolution : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_group_convolution(inputParams[conv],
                                                                                 ov::element::f32,
                                                                                 kernelSize,
                                                                                 strides,
                                                                                 padBegin,
                                                                                 padEnd,
                                                                                 dilation,
                                                                                 paddingType,
                                                                                 numOutChannels,
                                                                                 numOfGroups);
            }
            break;
        }
        case nodeType::groupConvolutionBackpropData : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_group_convolution_backprop_data(inputParams[conv],
                                                                                               ov::element::f32,
                                                                                               kernelSize,
                                                                                               strides,
                                                                                               padBegin,
                                                                                               padEnd,
                                                                                               dilation,
                                                                                               paddingType,
                                                                                               numOutChannels,
                                                                                               numOfGroups);
            }
            break;
        }
        default: {
            throw std::runtime_error("Subgraph concat test doesn't support this type of operation");
        }
    }
    for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
        convolutionNodes[conv]->get_rt_info() = getCPUInfo();
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{convolutionNodes[0], convolutionNodes[1]}, axis);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    function = std::make_shared<ov::Model>(results, inputParams, "convolutionConcat");
}

TEST_P(ConvConcatSubgraphTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, pluginTypeNode);
};

/* ============= Common Convolution Params ============= */
const ov::op::PadType paddingType{ov::op::PadType::EXPLICIT};
const size_t numOutChannels{32};
const int axis{1};

const ov::Shape inputShapes2D{1, 64, 16, 16};
const std::vector<size_t> kernelSize2D{3, 3};
const std::vector<size_t> strides2D{2, 2};
const std::vector<ptrdiff_t> padBegin2D{1, 1};
const std::vector<ptrdiff_t> padEnd2D{1, 1};
const std::vector<size_t> dilation2D{1, 1};
commonConvParams convParams2D = commonConvParams{kernelSize2D, strides2D, padBegin2D, padEnd2D, dilation2D, numOutChannels, paddingType, 1};
commonConvParams groupConvParams2D = commonConvParams{kernelSize2D, strides2D, padBegin2D, padEnd2D, dilation2D, numOutChannels, paddingType, 2};

const ov::Shape inputShapes3D{1, 64, 8, 16, 16};
const std::vector<size_t> kernelSize3D{3, 3, 3};
const std::vector<size_t> strides3D{2, 2, 2};
const std::vector<ptrdiff_t> padBegin3D{1, 1, 1};
const std::vector<ptrdiff_t> padEnd3D{1, 1, 1};
const std::vector<size_t> dilation3D{1, 1, 1};
commonConvParams convParams3D = commonConvParams{kernelSize3D, strides3D, padBegin3D, padEnd3D, dilation3D, numOutChannels, paddingType, 1};
commonConvParams groupConvParams3D = commonConvParams{kernelSize3D, strides3D, padBegin3D, padEnd3D, dilation3D, numOutChannels, paddingType, 2};

namespace Kernel_1x1 {

/* ============= Kernel_1x1 (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2DConv = {
    conv_sse42_2D_1x1,
    conv_avx2_2D_1x1,
    conv_avx512_2D_1x1
};

commonConvParams convParams2D1x1 = commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, dilation2D, numOutChannels, paddingType, 1};

const auto params2DConv = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams2D1x1),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2DConv)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D1x1, ConvConcatSubgraphTest, params2DConv, ConvConcatSubgraphTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams2DDeconv = {
    conv_avx2_2D_1x1,
    conv_avx512_2D_1x1
};

const auto params2DDeconv = ::testing::Combine(
    ::testing::Values(nodeType::convolutionBackpropData),
    ::testing::Values(convParams2D1x1),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2DDeconv)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D1x1, ConvConcatSubgraphTest, params2DDeconv, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace Kernel_1x1

namespace GroupConvolutionBackpropDataDWConcat {

/* ============= GroupConvolutionBackpropData (DW 2D) ============= */
commonConvParams dwDeconvParams2D = commonConvParams{kernelSize2D, strides2D, padBegin2D, padEnd2D, dilation2D, numOutChannels, paddingType, numOutChannels};
const ov::Shape inputShapesDW2D{1, 32, 16, 16};
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_sse42_dw_2D,
    conv_avx2_dw_2D,
    conv_avx512_dw_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(dwDeconvParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapesDW2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_DWGroupConvolutionBackpropData2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionBackpropDataDWConcat

namespace GroupConvolutionDWConcat {

/* ============= GroupConvolution (DW 2D) ============= */
commonConvParams dwConvParams2D = commonConvParams{kernelSize2D, strides2D, padBegin2D, padEnd2D, dilation2D, numOutChannels, paddingType, numOutChannels};
const ov::Shape inputShapesDW2D{1, 32, 16, 16};
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_sse42_dw_2D,
    conv_avx2_dw_2D,
    conv_avx512_dw_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(dwConvParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapesDW2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_DWGroupConvolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolution (DW 3D) ============= */
commonConvParams dwConvParams3D = commonConvParams{kernelSize3D, strides3D, padBegin3D, padEnd3D, dilation3D, numOutChannels, paddingType, numOutChannels};
const ov::Shape inputShapesDW3D{1, 32, 8, 16, 16};
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_sse42_dw_3D,
    conv_avx2_dw_3D,
    conv_avx512_dw_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(dwConvParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapesDW3D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_DWGroupConvolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionDWConcat

namespace ConvolutionBackpropDataConcat {

/* ============= ConvolutionBackpropData (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_ref_2D,
    // conv_gemm_2D,
    conv_avx512_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::convolutionBackpropData),
    ::testing::Values(convParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= ConvolutionBackpropData (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_ref_3D,
    // conv_gemm_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::convolutionBackpropData),
    ::testing::Values(convParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace ConvolutionBackpropDataConcat

namespace ConvolutionConact {

/* ============= Convolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_ref_2D,
    conv_gemm_2D,
    conv_sse42_2D,
    conv_avx2_2D,
    conv_avx512_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= Convolution (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_ref_3D,
    conv_gemm_3D,
    conv_avx2_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace ConvolutionConact

namespace GroupConvolutionConcat {

/* ============= GroupConvolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_ref_2D,
    conv_gemm_2D,
    conv_sse42_2D,
    conv_avx2_2D,
    conv_avx512_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolution (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_ref_3D,
    conv_gemm_3D,
    conv_avx2_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionConcat

namespace GroupConvolutionBackpropDataConcat {

/* ============= GroupConvolutionBackpropData (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_ref_2D,
    // conv_gemm_2D,
    conv_avx2_2D,
    conv_avx512_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(groupConvParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolutionBackpropData (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_ref_3D,
    // conv_gemm_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(groupConvParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionBackpropDataConcat

}  // namespace test
}  // namespace ov
