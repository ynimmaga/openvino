// Copyright (C) Intel Corporation
// Licensed under the MIT License

#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free
#else
#include <cstdlib>  // For posix_memalign and free
#endif

#include "ov_custom_allocator.h"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace onnxruntime {
namespace openvino_ep_plugin {

static void* AlignedAlloc(size_t size, size_t alignment) {
  if (size == 0) return nullptr;

  void* ptr;
#if _MSC_VER
  ptr = _aligned_malloc(size, alignment);
  if (ptr == nullptr)
    throw std::bad_alloc();
#else
  int ret = posix_memalign(&ptr, alignment, size);
  if (ret != 0)
    throw std::bad_alloc();
#endif
  return ptr;
}

static void AlignedFree(void* ptr) {
  if (ptr == nullptr) return;

#ifdef _WIN32
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

OVRTAllocator::OVRTAllocator(ov::Core& core, const OrtMemoryInfo& mem_info, const std::string& ov_device)
    : core_(core), memory_info_(&mem_info) {
  if (ov_device != "CPU") {
    remote_ctx_ = core_.get_default_context(ov_device);
  }

  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
}

void* OVRTAllocator::AllocImpl(OrtAllocator* allocator, size_t size) {
  auto* ov_allocator = static_cast<OVRTAllocator*>(allocator);

  if (!ov_allocator->remote_ctx_) {
    return AlignedAlloc(size, default_alignment_);
  } else {
    // Use OpenVINO remote tensor allocation for non-CPU devices
    ov::Tensor* tensor = new ov::Tensor(
        ov_allocator->remote_ctx_.value().create_host_tensor(ov::element::Type_t::u8, {size}));
    std::lock_guard<std::mutex> lock(ov_allocator->mutex_);
    ov_allocator->allocated_.insert({tensor->data(), tensor});
    return tensor->data();
  }
}

void OVRTAllocator::FreeImpl(OrtAllocator* allocator, void* p) {
  auto* ov_allocator = static_cast<OVRTAllocator*>(allocator);

  if (!ov_allocator->remote_ctx_) {
    AlignedFree(p);
  } else {
    // For non-CPU allocation, find and delete the tensor
    std::lock_guard<std::mutex> lock(ov_allocator->mutex_);
    auto it = ov_allocator->allocated_.find(p);
    if (it != ov_allocator->allocated_.end()) {
      delete it->second;
      ov_allocator->allocated_.erase(it);
    }
  }
}

const OrtMemoryInfo* OVRTAllocator::InfoImpl(const OrtAllocator* allocator) {
  return static_cast<const OVRTAllocator*>(allocator)->memory_info_;
}

}  // namespace openvino_ep_plugin
}  // namespace onnxruntime
