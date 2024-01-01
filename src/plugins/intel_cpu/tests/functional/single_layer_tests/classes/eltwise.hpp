// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<
        EltwiseTestParams,
        CPUSpecificParams,
        fusingSpecificParams,
        bool> EltwiseLayerCPUTestParamsSet;

class EltwiseLayerCPUTest : public testing::WithParamInterface<EltwiseLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestUtils::CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj);

protected:
    ov::Tensor generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

private:
    utils::EltwiseTypes eltwiseType;
};

namespace Eltwise {

const std::vector<ov::AnyMap>& additional_config();

const std::vector<ElementType>& netType();
const std::vector<ov::test::utils::OpType>& opTypes();
const std::vector<utils::EltwiseTypes>& eltwiseOpTypesBinInp();
const std::vector<utils::InputLayerType>& secondaryInputTypes();

const std::vector<utils::EltwiseTypes>& eltwiseOpTypesBinInp();
const std::vector<utils::EltwiseTypes>& eltwiseOpTypesDiffInp();
const std::vector<utils::EltwiseTypes>& eltwiseOpTypesBinDyn();

const std::vector<CPUSpecificParams>& cpuParams_4D();
const std::vector<std::vector<ov::Shape>>& inShapes_4D();
const std::vector<std::vector<InputShape>>& inShapes_4D_dyn_const();
const std::vector<InputShape>& inShapes_4D_dyn_param();
const std::vector<std::vector<ov::Shape>>& inShapes_4D_1D();
const std::vector<CPUSpecificParams> & cpuParams_4D_1D_Constant_mode();
const std::vector<CPUSpecificParams>& cpuParams_4D_1D_Parameter_mode();

const std::vector<CPUSpecificParams>& cpuParams_5D();
const std::vector<std::vector<ov::Shape>>& inShapes_5D();
const std::vector<std::vector<ov::Shape>>& inShapes_5D_1D();
const std::vector<InputShape>& inShapes_5D_dyn_const();
const std::vector<InputShape>& inShapes_5D_dyn_param();
const std::vector<CPUSpecificParams>& cpuParams_5D_1D_constant();
const std::vector<CPUSpecificParams>& cpuParams_5D_1D_parameter();

const std::vector<utils::EltwiseTypes>& eltwiseOpTypesI32();

const std::vector<bool>& enforceSnippets();

}  // namespace Eltwise
}  // namespace test
}  // namespace ov
