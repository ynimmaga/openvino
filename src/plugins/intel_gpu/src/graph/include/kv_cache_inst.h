// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/kv_cache.hpp"
#include "primitive_inst.h"
#include "variable.hpp"

namespace cldnn {

template <>
struct typed_program_node<kv_cache> : public typed_program_node_base<kv_cache> {
private:
    using parent = typed_program_node_base<kv_cache>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using kv_cache_node = typed_program_node<kv_cache>;

template<>
class typed_primitive_inst<kv_cache> : public typed_primitive_inst_base<kv_cache>, public memory_state::variable {
    using parent = typed_primitive_inst_base<kv_cache>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(kv_cache_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const kv_cache_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const kv_cache_node& node);

    static int32_t get_prealloc_iter_num() {
        return 128;
    }

    static void update_pad(layout& l, int64_t pad, int64_t sequence_axis_legacy) {
        const auto& dyn_pad_dims = l.data_padding.get_dynamic_pad_dims();
        const auto& lower_padd = l.data_padding.lower_size().sizes();
        auto upper_padd = l.data_padding.upper_size().sizes();
        upper_padd[sequence_axis_legacy] = pad;
        l.data_padding = padding(lower_padd, upper_padd, 0.f, dyn_pad_dims);
    }

    static int64_t get_sequence_axis_legacy(int64_t sequence_axis, size_t past_layout_rank) {
        auto sequence_axis_legacy = sequence_axis;
        if (sequence_axis_legacy < 0)
            sequence_axis_legacy = past_layout_rank + sequence_axis_legacy;
        if (sequence_axis_legacy >= 2) {
            auto spatial_axis = sequence_axis_legacy - 2;
            // Default and minimum number of dimensions is 4
            auto spatial_size = std::max<size_t>(past_layout_rank, 4) - 2;
            sequence_axis_legacy = spatial_size - spatial_axis - 1 + 2;
        }
        return sequence_axis_legacy;
    }

    static int64_t get_max_pad(const layout& target_layout, size_t buffer_size, int64_t legacy_sequence_axis, std::string target_name = "") {
        if (buffer_size == 0)
            return 0;
        const size_t total_elements = target_layout.count();
        const int64_t concat_axis_size = target_layout.get_tensor().sizes()[legacy_sequence_axis];
        const int64_t sequence_element_size = total_elements / concat_axis_size;
        const int64_t max_sequence_elements = buffer_size / sequence_element_size;
        auto max_pad = std::max<int64_t>(max_sequence_elements - concat_axis_size, 0);
        auto target_layout_name = (target_name != "") ? target_name : "target_layout";
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] " << target_name  << " : " << target_layout.to_string() << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] buffer size " << buffer_size << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] total_elements " << total_elements << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] concat_axis_size = " << concat_axis_size << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] sequence_element_size = " << sequence_element_size << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] max_sequence_elements = " << max_sequence_elements << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] max_pad (max_sequence_elements - concat_axis_size) = " << max_pad << std::endl;
        return max_pad;
    }

    typed_primitive_inst(network& network, const kv_cache_node& desc);
    typed_primitive_inst(network& network) : parent(network), memory_state::variable("") {}
};

using kv_cache_inst = typed_primitive_inst<kv_cache>;

} // namespace cldnn
