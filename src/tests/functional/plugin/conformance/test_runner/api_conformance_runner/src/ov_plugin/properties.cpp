// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

const std::vector<ov::AnyMap> inproperties = {
        {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(generate_ov_configs(inproperties))),
                        OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
        {},
        {ov::enable_profiling(true)},
};

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(targetDevice),
                ::testing::ValuesIn(default_properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckGetSupportedROMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(targetDevice),
                        ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::getROMandatoryProperties())),
        OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckGetSupportedROMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(targetDevice),
                        ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::getROOptionalProperties())),
        OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckSetSupportedRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(targetDevice),
                        ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWMandatoryPropertiesValues())),
        OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckSetSupportedRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(targetDevice),
                        ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWOptionalPropertiesValues())),
        OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckSetIncorrectRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(targetDevice),
                        ::testing::ValuesIn(OVCheckSetIncorrectRWMetricsPropsTests::getWrongRWMandatoryPropertiesValues())),
        OVCheckSetIncorrectRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckSetIncorrectRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(targetDevice),
                        ::testing::ValuesIn(OVCheckSetIncorrectRWMetricsPropsTests::getWrongRWOptionalPropertiesValues())),
        OVCheckSetIncorrectRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckChangePropComplieModleGetPropTests_DEVICE_ID,
        ::testing::Combine(
                ::testing::Values(targetDevice),
                ::testing::Values(ov::AnyMap({}))),
        OVCheckChangePropComplieModleGetPropTests_DEVICE_ID::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckChangePropComplieModleGetPropTests_InferencePrecision,
        ::testing::Combine(
                ::testing::Values(targetDevice),
                ::testing::Values(ov::AnyMap({}))),
        OVCheckChangePropComplieModleGetPropTests_InferencePrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckMetricsPropsTests_ModelDependceProps,
        ::testing::Combine(
                ::testing::Values(targetDevice),
                ::testing::ValuesIn(OVCheckMetricsPropsTests_ModelDependceProps::getModelDependcePropertiesValues())),
        OVCheckMetricsPropsTests_ModelDependceProps::getTestCaseName);

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin_mandatory, OVGetMetricPropsTest,
        ::testing::Values(targetDevice));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVGetMetricPropsOptionalTest,
        ::testing::Values(targetDevice));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin_mandatory, OVGetAvailableDevicesPropsTest,
        ::testing::Values(targetDevice));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVPropertiesDefaultSupportedTests,
        ::testing::Values(targetDevice));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin_remove_mandatory, OVBasicPropertiesTestsP,
        ::testing::ValuesIn(generate_ov_pairs_plugin_name_by_device()));
} // namespace
