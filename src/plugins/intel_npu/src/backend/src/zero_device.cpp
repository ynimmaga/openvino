// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_device.hpp"

#include <ze_api.h>

#include "intel_npu/al/itt.hpp"
#include "zero_executor.hpp"
#include "zero_infer_request.hpp"

using namespace intel_npu;

ZeroDevice::ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs)
    : _initStructs(initStructs),
      _graph_ddi_table_ext(_initStructs->getGraphDdiTable()),
      log("ZeroDevice", Logger::global().level()) {
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties",
                           zeDeviceGetProperties(_initStructs->getDevice(), &device_properties));

    // Query PCI information
    // Older drivers do not have this implementend. Linux driver returns NOT_IMPLEMENTED, while windows driver returns
    // zero values. If this is detected, we populate only device with ID from device_properties for backwards
    // compatibility
    pci_properties.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    ze_result_t retpci = zeDevicePciGetPropertiesExt(_initStructs->getDevice(), &pci_properties);
    if (ZE_RESULT_SUCCESS == retpci) {
        // win backwards compatibility
        if (pci_properties.address.device == 0) {
            log.warning("PCI information not available in driver. Falling back to deviceId");
            pci_properties.address.device = device_properties.deviceId;
        }
    } else if (ZE_RESULT_ERROR_UNSUPPORTED_FEATURE == retpci) {
        log.warning("PCI information not available in driver. Falling back to deviceId");
        // linux backwards compatibilty
        pci_properties.address.device = device_properties.deviceId;
    } else {
        OPENVINO_THROW("L0 zeDevicePciGetPropertiesExt result: ",
                       ze_result_to_string(retpci),
                       ", code 0x",
                       std::hex,
                       uint64_t(retpci),
                       " - ",
                       ze_result_to_description(retpci));
    }

    std::vector<ze_command_queue_group_properties_t> command_group_properties;
    uint32_t command_queue_group_count = 0;
    // Discover all command queue groups
    zeroUtils::throwOnFail(
        "zeDeviceGetCommandQueueGroupProperties",
        zeDeviceGetCommandQueueGroupProperties(_initStructs->getDevice(), &command_queue_group_count, nullptr));

    command_group_properties.resize(command_queue_group_count);

    for (auto& prop : command_group_properties) {
        prop.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        prop.pNext = nullptr;
    }

    zeroUtils::throwOnFail("zeDeviceGetCommandQueueGroupProperties",
                           zeDeviceGetCommandQueueGroupProperties(_initStructs->getDevice(),
                                                                  &command_queue_group_count,
                                                                  command_group_properties.data()));

    // Find the corresponding command queue group.
    _group_ordinal = zeroUtils::findGroupOrdinal(command_group_properties, device_properties);
}

std::shared_ptr<IExecutor> ZeroDevice::createExecutor(
    const std::shared_ptr<const NetworkDescription>& networkDescription,
    const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Device::createExecutor");
    return std::make_shared<ZeroExecutor>(_initStructs, networkDescription, config, _group_ordinal);
}

std::string ZeroDevice::getName() const {
//    KMD is setting usDeviceID from VpuFamilyID.h
#define NPU_3700_DEVICE_ID   0x6240
#define NPU_3720_P_DEVICE_ID 0x7D1D
#define NPU_3720_S_DEVICE_ID 0xAD1D

    std::string name;
    switch (device_properties.deviceId) {
    case NPU_3700_DEVICE_ID:
        name = "3700";
        break;
    case NPU_3720_P_DEVICE_ID:
    case NPU_3720_S_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU3720;
        break;
    default:
        name = "AUTO_DETECT";
    }

    return name;
}

std::string ZeroDevice::getFullDeviceName() const {
    return device_properties.name;
}

IDevice::Uuid ZeroDevice::getUuid() const {
    Uuid uuid{};
    static_assert(sizeof(device_properties.uuid.id) == uuid.uuid.size(),
                  "ze_device_uuid_t::id size doesn't match intel_npu::Uuid::uuid size");

    std::copy(std::begin(device_properties.uuid.id), std::end(device_properties.uuid.id), std::begin(uuid.uuid));

    return uuid;
}

uint32_t ZeroDevice::getSubDevId() const {
    return device_properties.subdeviceId;
}

uint32_t ZeroDevice::getMaxNumSlices() const {
    return device_properties.numSlices;
}

uint64_t ZeroDevice::getAllocMemSize() const {
    ze_graph_memory_query_t query{};
    zeroUtils::throwOnFail(
        "pfnQueryContextMemory",
        _graph_ddi_table_ext->pfnQueryContextMemory(_initStructs->getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query));
    return query.allocated;
}

uint64_t ZeroDevice::getTotalMemSize() const {
    ze_graph_memory_query_t query{};
    zeroUtils::throwOnFail(
        "pfnQueryContextMemory",
        _graph_ddi_table_ext->pfnQueryContextMemory(_initStructs->getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query));
    return query.total;
}

ov::device::PCIInfo ZeroDevice::getPciInfo() const {
    return ov::device::PCIInfo{pci_properties.address.domain,
                               pci_properties.address.bus,
                               pci_properties.address.device,
                               pci_properties.address.function};
}

std::shared_ptr<SyncInferRequest> ZeroDevice::createInferRequest(
    const std::shared_ptr<const ICompiledModel>& compiledModel,
    const std::shared_ptr<IExecutor>& executor,
    const Config& config) {
    return std::make_shared<ZeroInferRequest>(_initStructs, compiledModel, executor, config);
}
