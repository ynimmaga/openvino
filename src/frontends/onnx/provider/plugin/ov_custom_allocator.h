// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <unordered_map>
#include <mutex>
#include "openvino/runtime/core.hpp"

#include "ov_utils.h"

namespace onnxruntime {
namespace openvino_ep_plugin {

class OVRTAllocator : public OrtAllocator {
 public:
  static constexpr const char* allocator_name_ = "OpenVINO_shared";
  static constexpr size_t default_alignment_ = 4096;

  OVRTAllocator(ov::Core& core, const OrtMemoryInfo& mem_info, const std::string& ov_device);
  virtual ~OVRTAllocator() = default;

 private:
  static void* AllocImpl(OrtAllocator* allocator, size_t size);
  static void FreeImpl(OrtAllocator* allocator, void* p);
  static const OrtMemoryInfo* InfoImpl(const OrtAllocator* allocator);

  ov::Core& core_;
  std::optional<ov::RemoteContext> remote_ctx_;  // Only for NPU/GPU
  std::unordered_map<void*, ov::Tensor*> allocated_;
  const OrtMemoryInfo* memory_info_;
  std::mutex mutex_;
};

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
