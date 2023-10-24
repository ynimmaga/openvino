// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "loop_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include <vector>
#include <algorithm>

namespace cldnn {
namespace common {

// read scala value from data primitive
static int64_t read_scalar_value(memory::ptr mem, stream& stream) {
    int64_t trip_count = 0;
    const layout& prim_layout = mem->get_layout();

    switch (prim_layout.data_type) {
    case data_types::u8: {
        mem_lock<uint8_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i8: {
        mem_lock<int8_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i32: {
        mem_lock<int32_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    case data_types::i64: {
        mem_lock<int64_t> lock_prim_output{mem, stream};
        trip_count = *lock_prim_output.data();
        break;
    }
    default:
        OPENVINO_THROW("Invalid data type : ",  ov::element::Type(prim_layout.data_type).get_type_name());
    }
    return trip_count;
}

template<typename T>
static inline void validate_input_value(int64_t input) {
    OPENVINO_ASSERT((input >= std::numeric_limits<T>::min() && input <= std::numeric_limits<T>::max()),
                "Invalid data value : ", input);
}

static void write_scalar_value(memory::ptr mem, stream& stream, int64_t input) {
    const layout& prim_layout = mem->get_layout();

    switch (prim_layout.data_type) {
    case data_types::u8: {
        validate_input_value<uint8_t>(input);
        mem_lock<uint8_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<uint8_t>(input);
        break;
    }
    case data_types::i8: {
        validate_input_value<int8_t>(input);
        mem_lock<int8_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<int8_t>(input);
        break;
    }
    case data_types::i32: {
        validate_input_value<int32_t>(input);
        mem_lock<int32_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = static_cast<int32_t>(input);
        break;
    }
    case data_types::i64: {
        mem_lock<int64_t> lock_prim_output{mem, stream};
        lock_prim_output[0] = input;
        break;
    }
    default:
        OPENVINO_THROW("Invalid data type : ",  ov::element::Type(prim_layout.data_type).get_type_name());
    }
}

struct loop_impl : typed_primitive_impl<loop> {
    using parent = typed_primitive_impl<loop>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::loop_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<loop_impl>(*this);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    loop_impl() : parent() {}

    loop_impl(const loop_impl& other) : typed_primitive_impl<loop>(other),
        _back_edges(other._back_edges) {}

    explicit loop_impl(const loop_node& node) {
        set_node_params(node);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<loop>());
        const auto& node = arg.as<loop>();
        _back_edges = node.get_back_edges();
    }

    void set_memory_in_body_network(cldnn::network::ptr body_network,
                    const std::shared_ptr<cldnn::primitive_inst>& inst, memory::ptr mem) const {
        if (inst->is_input()) {
            body_network->set_input_data(inst->id(), mem);
        } else if (inst->is_output()) {
            body_network->set_output_memory(inst->id(), mem);
        } else {
            inst->set_output_memory(mem, false);
        }
    }

    std::vector<event::ptr> handle_buffers_for_next_iteration(const loop_inst::backedge_memory_mapping& mapping,
                                                    network::ptr body_network, int64_t iter, bool is_dynamic) const {
        std::vector<event::ptr> event_vec;
        OPENVINO_ASSERT(iter >= 0, "iteration should not be negative : ", iter);
        if (mapping.type == loop_inst::backedge_memory_mapping::CONCAT_OUTPUT) {
            if (iter == 0) {
                set_memory_in_body_network(body_network, mapping.to_primitive, mapping.initial_mem);
            } else if (iter > 0) {
                if (is_dynamic) {
                    auto from_id = mapping.from_primitive->id();
                    if (body_network->has_event(from_id)) {
                        auto ev = body_network->get_primitive_event(from_id);
                        if (ev) ev->wait();
                    }
                    // In dynamic model, just copy data from inner body output to inner body input in back_edges.
                    memory::ptr mem1 = mapping.to_primitive->output_memory_ptr();
                    memory::ptr mem2 = mapping.from_primitive->output_memory_ptr();
                    auto ev = mem1->copy_from(body_network->get_stream(), *(mem2));
                    if (ev) event_vec = {ev};
                } else {
                    auto mem = mapping.concat_mem_mapping->get_sliced_mems().at(iter - 1);
                    set_memory_in_body_network(body_network, mapping.to_primitive, mem);
                }
            }
        } else if (mapping.type ==  loop_inst::backedge_memory_mapping::SINGLE_SHARED) {
            if (iter == 0) {
                if (mapping.from_mem != nullptr) {
                    auto ev = mapping.from_mem->copy_from(body_network->get_stream(), *(mapping.initial_mem));
                    if (ev) event_vec = {ev};
                }
            } else {
                // In dynamic model, output memory is not defined before execution.
                // After body network execution, replace input memory from initial_mem(external input memory) to output memory.
                if (mapping.from_mem == nullptr) {
                    mapping.from_mem = mapping.from_primitive->output_memory_ptr();
                    OPENVINO_ASSERT(mapping.from_mem != nullptr, "from_mem should not be null");
                    set_memory_in_body_network(body_network, mapping.to_primitive, mapping.from_mem);
                }
            }
        } else if (mapping.type ==  loop_inst::backedge_memory_mapping::SINGLE) {
            memory::ptr mem1 = mapping.to_primitive->output_memory_ptr();
            if (iter == 0) {
                auto ev = mem1->copy_from(body_network->get_stream(), *(mapping.initial_mem));
                if (ev) event_vec = {ev};
            } else {
                if (is_dynamic) {
                    // In dynamic model, do not set memory buffer between input and output in inner body network.
                    // Just copy data from input buffer memory to output buffer memory.
                    auto from_id = mapping.from_primitive->id();
                    if (body_network->has_event(from_id)) {
                        auto ev = body_network->get_primitive_event(from_id);
                        if (ev) ev->wait();
                    }
                    memory::ptr mem2 = mapping.from_primitive->output_memory_ptr();
                    auto ev = mem1->copy_from(body_network->get_stream(), *(mem2));
                    if (ev) event_vec = {ev};
                } else {
                    // In static model, swap memory buffer between output and input in inner body network
                    memory::ptr mem2 = mapping.from_primitive->output_memory_ptr();
                    set_memory_in_body_network(body_network, mapping.to_primitive, std::move(mem2));
                    set_memory_in_body_network(body_network, mapping.from_primitive, std::move(mem1));
                }
            }
        }
        return event_vec;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, loop_inst& instance) override {
        const auto& impl_params = instance.get_impl_params();
        const auto& primitive = impl_params->typed_desc<loop>();
        auto& outer_network = instance.get_network();
        auto& stream = outer_network.get_stream();

        const auto max_num_iterations = primitive->max_num_iterations;
        auto body_network = instance.get_body_network();
        int64_t current_iteration_idx = 0;

        auto ev = stream.create_user_event(false);

        OPENVINO_ASSERT(!primitive->num_iteration_id.empty(), "loop operation should have num_iteration_id");

        //////////////////////////////////////////
        // memory pointers for outer network
        //////////////////////////////////////////
        // read trip_count from outer network
        int64_t trip_count = -1;
        if (!primitive->trip_count_id.empty()) {
            memory::ptr trip_count_mem = outer_network.get_primitive(primitive->trip_count_id)->output_memory_ptr();
            trip_count = read_scalar_value(std::move(trip_count_mem), stream);
        } else {
            trip_count = max_num_iterations;
        }

        // read initial execution condition from outer network
        int64_t execution_condition = 1;
        if (!primitive->first_execution_condition_id.empty()) {
            memory::ptr first_execution_condition_mem = outer_network.get_primitive(primitive->first_execution_condition_id)->output_memory_ptr();
            execution_condition = read_scalar_value(first_execution_condition_mem, stream);
        }

        // When execution_condition is false or trip_count is zero, return execute_impl without any body_network execution.
        if (!execution_condition || trip_count == 0) {
            // Update num_iterations (actual number of iterations)
            memory::ptr num_actual_iterations_mem = outer_network.get_primitive(primitive->num_iteration_id)->output_memory_ptr();
            write_scalar_value(num_actual_iterations_mem, stream, current_iteration_idx);

            instance.update_output_layout();
            ev->set();
            return ev;
        }

        //////////////////////////////////////////
        // memory pointers for body network
        //////////////////////////////////////////
        // shortcut of execution_condition memory in body network
        memory::ptr body_execution_condition_mem = nullptr;
        if (!primitive->body_execution_condition_id.empty()) {
            body_execution_condition_mem = body_network->get_primitive(primitive->body_execution_condition_id)->output_memory_ptr();
        }

        // shortcut of current_iteration memory in body network
        if (!primitive->body_current_iteration_id.empty()) {
            memory::ptr body_current_iteration_mem = body_network->get_primitive(primitive->body_current_iteration_id)->output_memory_ptr();
            write_scalar_value(body_current_iteration_mem, body_network->get_stream(), 0);
        }

        const auto is_dynamic = instance.is_dynamic();
        if (is_dynamic) {
            instance.update_shape();
            if (instance.shape_changed()) {
                instance.preproc_memories_done = false;
                instance.reset_memory();
            }
        }

        if (!instance.preproc_memories_done) {
            instance.preprocess_output_memory(trip_count);
            instance.preprocess_input_memory(trip_count);
            instance.preprocess_backedge_memory();
            instance.preproc_memories_done = true;
        }

        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        const auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;
        const auto& backedge_memory_mappings = instance.backedge_memory_mappings;

        // If there are concatenated_output_mem_mappings or backedge_memory_mappings we need to wait for
        // previous tasks before accessing memory in get_sliced_mem() and setup_iteration() functions
        if (!concatenated_input_mem_mappings.empty() || !backedge_memory_mappings.empty()) {
            for (auto& e : events) {
                e->wait();
            }
        }

        // Set sliced input data
        for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
            const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
            memory::ptr mem = concatenated_input->get_sliced_mem(0);
            OPENVINO_ASSERT(mem != nullptr, instance.id(), "sliced input memory of loop is not allocated properly");
            body_network->set_input_data(concatenated_input->sliced_data_prim->id(), mem);
        }

        std::vector<event::ptr> all_events;
        std::vector<event::ptr> loop_carried_dep(events.begin(), events.end());
        while (((trip_count <= 0) || (current_iteration_idx < trip_count)) && execution_condition) {
            // Copy & Set sliced input memory
            for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
                const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
                memory::ptr mem = concatenated_input->get_sliced_mem(current_iteration_idx);
                OPENVINO_ASSERT(mem != nullptr, instance.id(), " sliced input memory of loop is not allocated properly");
                concatenated_input->sliced_data_prim->set_output_memory(mem);
            }

            // Set backedges and output memory
            for (auto& backedge_memory_mapping : backedge_memory_mappings) {
                auto event_vec = handle_buffers_for_next_iteration(backedge_memory_mapping, body_network, current_iteration_idx, is_dynamic);
                for (auto ev : event_vec) {
                    loop_carried_dep.push_back(ev);
                }
            }

            if (!is_dynamic) {
                // Set sliced output memory for static shape model
                // because body network generate output memory during the body network execution in dynamic model
                for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
                    concat_output_mem_mapping->setup_sliced_output_memory(current_iteration_idx);
                }
            }

            // execute body network
            body_network->execute(loop_carried_dep);

            loop_carried_dep.clear();
            for (const auto& backedge : _back_edges) {
                event::ptr body_event;
                if (body_network->has_event(backedge.from)) {
                    body_event = body_network->get_primitive_event(backedge.from);
                    loop_carried_dep.emplace_back(body_event);
                }
            }

            // Collect output events for waiting for all iterations finishing
            for (auto& out : body_network->get_outputs()) {
                auto output_id = out->id();
                if (body_network->has_event(output_id)) {
                    auto output_event = body_network->get_primitive_event(output_id);
                    all_events.push_back(output_event);
                }
            }

            // Store output of sliced_data_prim to sliced mems vector
            // After execution of body network, sliced_data_prim will has output memory buffer
            // current memory buffer move to sliced_mems and new memory buffer will be allocated in sliced_data_prim
            if (is_dynamic) {
                for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
                    auto sliced_data_prim = concat_output_mem_mapping->sliced_data_prim;
                    auto output_mem_ptr = sliced_data_prim->output_memory_ptr();

                    auto sliced_id = sliced_data_prim->id();
                    if (body_network->has_event(sliced_id)) {
                        auto ev = body_network->get_primitive_event(sliced_id);
                        if (ev) ev->wait();
                    }
                    memory::ptr new_sliced_mem = concat_output_mem_mapping->get_or_create_sliced_mem(current_iteration_idx,
                                                                                                output_mem_ptr->get_layout());
                    auto ev = new_sliced_mem->copy_from(body_network->get_stream(), *output_mem_ptr);
                    if (ev) {
                        loop_carried_dep.push_back(ev);
                        all_events.push_back(ev);
                    }
                }
            }

            // execution condition is the result of body network execution
            if (body_execution_condition_mem != nullptr) {
                auto execution_id = primitive->body_execution_condition_id;
                if (body_network->has_event(execution_id)) {
                    auto ev = body_network->get_primitive_event(execution_id);
                    if (ev) ev->wait();
                }
                execution_condition = read_scalar_value(body_execution_condition_mem, body_network->get_stream());
            }
            GPU_DEBUG_IF(!execution_condition) {
                GPU_DEBUG_LOG << "body_exec_condition is false at "<< current_iteration_idx << " iterations" << std::endl;
            }

            current_iteration_idx++;
        }

        // Reset network and wait for all collected events
        body_network->reset_execution(false);
        stream.wait_for_events(all_events);

        // Update actual num iteration
        // update num_iterations (actual number of iterations)
        memory::ptr num_actual_iterations_mem = outer_network.get_primitive(primitive->num_iteration_id)->output_memory_ptr();
        write_scalar_value(num_actual_iterations_mem, stream, current_iteration_idx);
        GPU_DEBUG_LOG << "current_iteration(" << primitive->num_iteration_id << ", "
                        << num_actual_iterations_mem << ")  : " << current_iteration_idx << std::endl;

        if (is_dynamic)
            instance.update_output_layout();
        instance.postprocess_output_memory(is_dynamic);

        ev->set();
        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const loop_node& arg, const kernel_impl_params&) {
        return make_unique<loop_impl>(arg);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _back_edges;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _back_edges;
    }

private:
    std::vector<cldnn::loop::backedge_mapping> _back_edges;
};

namespace detail {
attach_loop_common::attach_loop_common() {
    implementation_map<loop>::add(impl_types::common,
                                    shape_types::dynamic_shape,
                                    loop_impl::create,
                                    {},
                                    {});
    implementation_map<loop>::add(impl_types::common, loop_impl::create, {});
}
}  // namespace detail

}  // namespace common
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::loop_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::loop)
