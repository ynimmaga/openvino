// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for internal properties that are passed from one plugin to another
 * @file openvino/runtime/internal_properties.hpp
 */

#pragma once

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {

namespace internal {
/**
 * @brief Read-only property to get a std::vector<PropertyName> of supported internal properties.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::vector<PropertyName>, PropertyMutability::RO> supported_properties{
    "INTERNAL_SUPPORTED_PROPERTIES"};

/**
 * @brief Read-only property to get a std::vector<PropertyName> of properties
 * which should affect the hash calculation for model cache
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::vector<PropertyName>, PropertyMutability::RO> caching_properties{"CACHING_PROPERTIES"};

/**
 * @brief Allow to create exclusive_async_requests with one executor
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RW> exclusive_async_requests{"EXCLUSIVE_ASYNC_REQUESTS"};

/**
 * @brief the property for setting of required device for which config to be updated
 * values: device id starts from "0" - first device, "1" - second device, etc
 * note: plugin may have different devices naming convention
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::string, PropertyMutability::WO> config_device_id{"CONFIG_DEVICE_ID"};

/**
 * @brief The name for setting CPU affinity per thread option.
 *
 * It is passed to Core::get_property()
 *
 * The following options are implemented only for the TBB as a threading option
 * ov::threading::IStreamsExecutor::ThreadBindingType::NUMA (pinning threads to NUMA nodes, best for real-life,
 * contented cases) on the Windows and MacOS* this option behaves as YES
 * ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE (let the runtime to do pinning to the cores types,
 * e.g. prefer the "big" cores for latency tasks) on the hybrid CPUs this option is default
 *
 * Also, the settings are ignored, if the OpenVINO compiled with OpenMP and any affinity-related OpenMP's
 * environment variable is set (as affinity is configured explicitly)
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<ov::threading::IStreamsExecutor::ThreadBindingType, PropertyMutability::RW> cpu_bind_thread{
    "CPU_BIND_THREAD"};

/**
 * @brief Limit \#threads that are used by IStreamsExecutor to execute `parallel_for` calls
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<size_t, PropertyMutability::RW> threads_per_stream{"THREADS_PER_STREAM"};

/**
 * @brief It contains compiled_model_runtime_properties information to make plugin runtime can check whether it is
 * compatible with the cached compiled model, the result is returned by get_property() calling.
 *
 * The information details are defined by plugin itself, each plugin may require different runtime contents.
 * For example, CPU plugin will contain OV version, while GPU plugin will contain OV and GPU driver version, etc.
 * Core doesn't understand its content and only read it from plugin and write it into blob header.
 *
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::string, PropertyMutability::RO> compiled_model_runtime_properties{
    "COMPILED_MODEL_RUNTIME_PROPERTIES"};

/**
 * @brief Check whether the attached compiled_model_runtime_properties is supported by this device runtime.
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RO> compiled_model_runtime_properties_supported{
    "COMPILED_MODEL_RUNTIME_PROPERTIES_SUPPORTED"};

}  // namespace internal
OPENVINO_DEPRECATED(
    "This property is deprecated and will be removed soon. Use ov::internal::caching_properties instead of it.")
constexpr auto caching_properties = internal::caching_properties;
OPENVINO_DEPRECATED(
    "This property is deprecated and will be removed soon. Use ov::internal::exclusive_async_requests instead of it.")
constexpr auto exclusive_async_requests = internal::exclusive_async_requests;
}  // namespace ov
