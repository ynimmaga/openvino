// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief EXPERIMENTAL. Lets an application run its own GPU kernel (e.g. one built by Triton) as a node
 * inside a model compiled for the GPU plugin, on the plugin's own queue.
 *
 * An op carrying a shared_ptr<ExternalOpCallback> in its rt_info under external_op_rt_info_key is not
 * lowered by the plugin. At that node's position in the graph the plugin calls enqueue() with its
 * native in-order queue and the node's input/output buffers; the callback appends its work there.
 * Work enqueued before the node is ordered before it by the queue, and the plugin orders everything
 * after it, so no host synchronization is involved.
 *
 * @file external_op.hpp
 */
#pragma once

#include <memory>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_gpu {

/// A node input or output as the plugin allocated it: a device-accessible pointer (USM on the
/// Level Zero runtime) to the first element, its element type, its actual (static) shape for this
/// execution, and per-dimension strides in elements. Strides are not necessarily dense: a view the
/// plugin keeps in place (e.g. a slice of a larger buffer) is described rather than copied.
struct ExternalOpTensor {
    void* data = nullptr;
    ov::element::Type type;
    ov::Shape shape;
    std::vector<int64_t> strides;
};

class ExternalOpCallback {
public:
    virtual ~ExternalOpCallback() = default;

    /// @param native_queue the plugin's in-order queue: ze_command_list_handle_t (immediate command
    /// list) on the Level Zero runtime. The device and context can be queried from it.
    virtual void enqueue(void* native_queue,
                         const std::vector<ExternalOpTensor>& inputs,
                         const std::vector<ExternalOpTensor>& outputs) = 0;
};

/// rt_info key holding std::shared_ptr<ExternalOpCallback> on an op the GPU plugin should not lower.
static constexpr const char external_op_rt_info_key[] = "intel_gpu_external_op";

}  // namespace intel_gpu
}  // namespace ov
