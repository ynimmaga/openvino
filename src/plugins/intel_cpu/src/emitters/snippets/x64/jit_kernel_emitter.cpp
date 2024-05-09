// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitter.hpp"
#include "snippets/utils.hpp"


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

inline static std::vector<Reg64> transform_idxs_to_regs(const std::vector<size_t>& idxs) {
    std::vector<Reg64> regs(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});
    return regs;
}

inline static std::vector<size_t> transform_snippets_regs_to_idxs(const std::vector<snippets::Reg>& regs) {
    std::vector<size_t> idxs(regs.size());
    std::transform(regs.cbegin(), regs.cend(), idxs.begin(), [](const snippets::Reg& reg) { return reg.idx; });
    return idxs;
}

jit_snippets_call_args::~jit_snippets_call_args() {
    delete[] loop_args;
}

void jit_snippets_call_args::register_loops(const std::vector<loop_args_t>& loops) {
    num_loops = loops.size();
    loop_args = new loop_args_t[num_loops];
    std::copy(loops.begin(), loops.end(), loop_args);
}

jit_snippets_call_args::loop_args_t::loop_args_t(int64_t work_amount, const std::vector<int64_t>& ptr_increments,
                                                 const std::vector<int64_t>& finalization_offsets)
    : m_work_amount(work_amount) {
    OV_CPU_JIT_EMITTER_ASSERT(ptr_increments.size() == finalization_offsets.size(), "Inconsistent sizes of ptr_increments and finalization_offsets");
    m_num_data_ptrs = static_cast<int64_t>(ptr_increments.size());
    init_pointers_and_copy_data(m_num_data_ptrs, ptr_increments.data(), finalization_offsets.data());
}

jit_snippets_call_args::loop_args_t::loop_args_t(const loop_args_t& other)
    : m_work_amount(other.m_work_amount), m_num_data_ptrs(other.m_num_data_ptrs) {
    init_pointers_and_copy_data(m_num_data_ptrs, other.m_ptr_increments, other.m_finalization_offsets);
}

jit_snippets_call_args::loop_args_t::~loop_args_t() {
    delete[] m_ptr_increments;
    delete[] m_finalization_offsets;
}

jit_snippets_call_args::loop_args_t& jit_snippets_call_args::loop_args_t::operator=(loop_args_t other) {
    swap(*this, other);
    return *this;
}

void jit_snippets_call_args::loop_args_t::init_pointers_and_copy_data(const int64_t num_elements, const int64_t* ptr_increments,
                                                                      const int64_t* finalization_offsets) {
    const size_t chunk_size = num_elements * sizeof(int64_t);
    m_ptr_increments = new int64_t[num_elements];
    m_finalization_offsets = new int64_t[num_elements];
    std::memcpy(m_ptr_increments, ptr_increments, chunk_size);
    std::memcpy(m_finalization_offsets, finalization_offsets, chunk_size);
}

void swap(jit_snippets_call_args::loop_args_t& first, jit_snippets_call_args::loop_args_t& second) {
    std::swap(first.m_work_amount, second.m_work_amount);
    std::swap(first.m_num_data_ptrs, second.m_num_data_ptrs);
    std::swap(first.m_ptr_increments, second.m_ptr_increments);
    std::swap(first.m_finalization_offsets, second.m_finalization_offsets);
}

jit_kernel_emitter::jit_kernel_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_container_emitter(h, isa), reg_runtime_params_idx(abi_param1.getIdx()) {
    const auto kernel = ov::as_type_ptr<snippets::op::Kernel>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "invoked with invalid op argument");
    OV_CPU_JIT_EMITTER_ASSERT(!kernel->region.empty(), "invoked with empty body");
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    num_inputs = 0;
    num_outputs = 0;
    const auto& io_exprs = body.get_IO_ops();
    for (const auto& expr : io_exprs) {
        switch (expr->get_type()) {
            case snippets::lowered::IOExpression::io_type::INPUT: {
                num_inputs++;
                break;
            }
            case snippets::lowered::IOExpression::io_type::OUTPUT: {
                num_outputs++;
                break;
            } default : {
                OV_CPU_JIT_EMITTER_THROW("detected unsupported io_type");
            }
        }
        mem_access_exprs.push_back(expr);
    }
    std::set<size_t> unique_buffers;
    for (const auto& expr : body) {
        if (const auto buffer = ov::as_type_ptr<snippets::op::Buffer>(expr->get_node())) {
            const auto buffer_id = buffer->get_id();
            if (unique_buffers.count(buffer_id) == 0) {
                mem_access_exprs.push_back(expr);
                unique_buffers.insert(buffer_id);
            }
        } else {
            if (std::find(io_exprs.cbegin(), io_exprs.cend(), expr) == io_exprs.cend())
                general_exprs.emplace_back(expr);
        }
    }
    num_unique_buffers = unique_buffers.size();
}

void jit_kernel_emitter::init_reg_pools(const std::set<size_t>& gpr_blacklist, const std::set<size_t>& vec_blacklist) {
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 16; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 15 - i;
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                  [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    std::set<size_t> gprs_blacklist_extended{Xbyak::Operand::RSP, Xbyak::Operand::RBP};
    gprs_blacklist_extended.insert(gpr_blacklist.begin(), gpr_blacklist.end());
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, gprs_blacklist_extended);
    remove_regs_from_pool(vec_regs_pool, vec_blacklist);
}

void jit_kernel_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_kernel_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty() && out.empty(), ": expects 0 registers on input and output");
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    OV_CPU_JIT_EMITTER_ASSERT(data_ptr_regs_idx.size() == num_params,
                              "number of inputs and outputs is inconsistent with the number of allocated registers ", num_params,
                              " data_ptr_regs_idx.size() = ", data_ptr_regs_idx.size());
}

void jit_kernel_emitter::init_body_regs(const std::set<size_t>& kernel_regs,
                                        const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) {
    // Initialize pools of gp and vec registers
    // Reserve kernel regs (abi_param1 and, if there is, abi_param2), since they'll be used to pass runtime call args to kernel
    init_reg_pools(kernel_regs, {});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);

    // Note that we can't use kernel_regs to store data pointers because
    // these regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);

    gpr_map_pool.second.insert(gpr_map_pool.second.end(), pool_gpr_idxs.cbegin(), pool_gpr_idxs.cend());
    vec_map_pool.second.insert(vec_map_pool.second.end(), pool_vec_idxs.cbegin(), pool_vec_idxs.cend());
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}

void jit_kernel_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    h->preamble();

    auto data_ptr_regs = transform_idxs_to_regs(data_ptr_regs_idx);

    init_data_pointers(data_ptr_regs);
    for (const auto& expression : body) {
        const auto reg_info = expression->get_reg_info();
        auto in_regs = transform_snippets_regs_to_idxs(reg_info.first);
        auto out_regs = transform_snippets_regs_to_idxs(reg_info.second);
        const auto& emitter = expression->get_emitter();
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }

    h->postamble();
}

jit_kernel_static_emitter::jit_kernel_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr), reg_indexes_idx(abi_param2.getIdx()) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "jit_kernel_static_emitter expectes KernelStatic expression");
    master_shape = body.get_master_shape();
    io_shapes.reserve(num_inputs + num_outputs);
    io_data_layouts.reserve(num_inputs + num_outputs);
    io_data_sizes.reserve(num_inputs + num_outputs);
    const auto& io_exprs = body.get_IO_ops();
    for (const auto& expr : io_exprs) {
        snippets::lowered::PortDescriptorPtr desc = nullptr;
        element::Type etype;
        switch (expr->get_type()) {
            case snippets::lowered::IOExpression::io_type::INPUT: {
                // input->shape changing ops->load
                const auto& shape_infer_seq = ov::snippets::utils::get_first_child_shape_infer_expr_seq(expr);
                const auto& mem_desc_expr = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
                auto consumer_inputs = mem_desc_expr->get_output_port_connector(0)->get_consumers();
                for (const auto& child_input : consumer_inputs) {
                    const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(child_input.get_expr()->get_node());
                    if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                        desc = child_input.get_descriptor_ptr();
                        break;
                    }
                }
                etype = mem_desc_expr->get_node()->get_output_element_type(0);
                break;
            }
            case snippets::lowered::IOExpression::io_type::OUTPUT: {
                // store->shape changing ops->result
                const auto& shape_infer_seq = ov::snippets::utils::get_first_parent_shape_infer_expr_seq(expr);
                const auto& mem_desc_expr = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
                desc = mem_desc_expr->get_input_port_connector(0)->get_source().get_descriptor_ptr();
                etype = mem_desc_expr->get_node()->get_input_element_type(0);
                break;
            } default : {
                OPENVINO_THROW("Kernel detected unsupported io_type");
            }
        }
        io_shapes.push_back(desc->get_shape());
        io_data_layouts.push_back(desc->get_layout());
        io_data_sizes.push_back(etype.size());
    }
    // Note: plugin can prepend master shape with 1 to facilitate parallel execution (usually up to 6D tensor)
    //       so we have to reproduce this behavior here
    master_shape.insert(master_shape.begin(), jcp.parallel_executor_ndims - master_shape.size(), 1);

    // - Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    // - However we can use reg_indexes_idx for non memory access operations
    //   since we won't need them after offsets calculation
    init_body_regs({reg_indexes_idx, reg_runtime_params_idx}, {}, {reg_indexes_idx});
}

void jit_kernel_static_emitter::init_data_pointers(const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    Xbyak::Reg64 reg_indexes = Xbyak::Reg64(static_cast<int>(reg_indexes_idx));
    Xbyak::Reg64 reg_runtime_params = Xbyak::Reg64(static_cast<int>(reg_runtime_params_idx));

    const auto num_params = num_inputs + num_outputs;
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = master_shape.size() - 1;
    std::vector<std::vector<size_t>> data_offsets(num_params, std::vector<size_t>{});
    auto offset_calculation = [=](const std::vector<size_t>& shape, const std::vector<size_t>& layout, const size_t data_size, bool is_input) {
        // Strides represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    strides: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    strides: s1*s3, s3,       0,  1
        std::vector<size_t> strides(shape.size());
        size_t dim_step = 1;
        strides[shape.size() - 1] = 1;
        for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
            dim_step *= shape[k+1];
            strides[k] = shape[k] != 1 ? dim_step * data_size : 0;
        }
        // Note: this is an extra copy, but let's keep it for clarity
        if (!layout.empty()) {
            std::vector<size_t> reordered_strides(strides.size());
            for (size_t i = 0; i < layout.size(); i++) {
                const auto& src_idx = is_input ? layout[i] : i;
                const auto& dst_idx = is_input ? i : layout[i];
                reordered_strides[dst_idx] = strides[src_idx];
            }
            strides = std::move(reordered_strides);
        }
        // the last stride is ignored, since the entire last dim is processed by kernel
        // and no parallel_for data_ptr offsets can be applied in this case
        strides.pop_back();
        // actual offset size might be larger that the shape size due to 6D scheduling
        strides.insert(strides.begin(), offset_rank - strides.size(), 0);

        return strides;
    };
    for (size_t i = 0; i < num_params; i++) {
        data_offsets[i] = offset_calculation(io_shapes[i],  io_data_layouts[i], io_data_sizes[i], i < num_inputs);
    }
    // master_shape size must be valid in both static and dynamic cases
    std::function<void(Reg64, const std::vector<size_t>&, Reg64)> init_ptr_with_offset;
    init_ptr_with_offset = [&](Reg64 pointer, const std::vector<size_t>& offsets, Reg64 reg_tmp) {
        for (size_t j = 0; j < offset_rank; j++) {
            if (master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    const auto spare_corruptable_gpr = std::find_if(gp_regs_pool.begin(), gp_regs_pool.end(),
                                                   [this](size_t reg) {
                                                        return reg != reg_indexes_idx && reg != reg_runtime_params_idx;
                                                   });
    const bool last_iter_explicitly = spare_corruptable_gpr == gp_regs_pool.end();
    Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs[num_params - 1] : Reg64(static_cast<int>(*spare_corruptable_gpr));
    // Vector "data_ptr_regs" is sorted by abstract regs.
    // It means that the vector contains the physical registers in order [src, .., src, dst, .., dst, buffer]
    // So we can initialize buffer register firstly as last value of vector "data_ptr_regs"
    // NOTE: Snippets Buffer Scratchpad has the common data pointer for all Buffers (even with different ID).
    //       The accessing memory is covered by correct offsets in each Buffer and the corresponding MemoryAccess ops
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_runtime_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_runtime_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_runtime_params to pass runtime args to LoopScheduler, so we have to
    //     push a reg on the stack, and restore it value afterward
    if (last_iter_explicitly) {
        h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_runtime_params;
        // can corrupt reg_runtime_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
}

jit_kernel_dynamic_emitter::jit_kernel_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelDynamic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "expectes KernelDynamic expression");

    // - Reserve abi_param1, since it wll be used to pass runtime call args to all dynamic emitters that needs runtime args
    // - We cannot assign this register to the body emitters since runtime params MUST be valid during whole execution
    //   for all dynamic emitters
    init_body_regs({reg_runtime_params_idx});
}

void jit_kernel_dynamic_emitter::init_data_pointers(const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    Xbyak::Reg64 reg_runtime_params = Xbyak::Reg64(static_cast<int>(reg_runtime_params_idx));

    const auto num_params = num_inputs + num_outputs;
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_runtime_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    for (size_t i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
    }
}

}   // namespace intel_cpu
}   // namespace ov
