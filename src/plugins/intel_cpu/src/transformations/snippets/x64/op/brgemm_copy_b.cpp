// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"
#include "snippets/op/buffer.hpp"

#include "brgemm_copy_b.hpp"

#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const Type type,
                                    const size_t offset_in, const size_t offset_out0, const size_t offset_out1,
                                    std::vector<size_t> layout_input, const size_t blk_size_k, const size_t blk_size_n)
    : snippets::modifier::MemoryAccess(1, type == Type::WithCompensations ? 2 : 1), op::Op({x}),
      m_type(type), m_src_type(src_type) {
    set_output_size(type == Type::WithCompensations ? 2 : 1);
    set_input_port_descriptor({0, offset_in}, 0);
    set_output_port_descriptor({0, offset_out0}, 0);
    if (is_with_compensations()) {
        set_output_port_descriptor({0, offset_out1}, 1);
    }
    compute_block_size_values(blk_size_k, blk_size_n);
    custom_constructor_validate_and_infer_types(std::move(layout_input));
    m_brgemmVNNIFactor = 4 / m_src_type.size();
    m_inner_n_block = 16 * m_brgemmVNNIFactor;
}

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const Type type,
                                    const PortDescriptor& desc_in0, const PortDescriptor& desc_out0, const PortDescriptor& desc_out1,
                                    std::vector<size_t> layout_input, const size_t blk_size_k, const size_t blk_size_n)
    : snippets::modifier::MemoryAccess(1, type == Type::WithCompensations ? 2 : 1), op::Op({x}),
      m_type(type), m_src_type(src_type) {
    set_output_size(type == Type::WithCompensations ? 2 : 1);
    set_input_port_descriptor(desc_in0, 0);
    set_output_port_descriptor(desc_out0, 0);
    if (is_with_compensations()) {
        set_output_port_descriptor(desc_out1, 1);
    }
    compute_block_size_values(blk_size_k, blk_size_n);
    custom_constructor_validate_and_infer_types(std::move(layout_input));
    m_brgemmVNNIFactor = 4 / m_src_type.size();
    m_inner_n_block = 16 * m_brgemmVNNIFactor;
}

bool BrgemmCopyB::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(BrgemmRepack_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    visitor.on_attribute("src_type", m_src_type);
    return true;
}

void BrgemmCopyB::custom_constructor_validate_and_infer_types(std::vector<size_t> layout_input) {
    INTERNAL_OP_SCOPE(BrgemmRepack_ctor_validate_and_infer_types);
    // During ctor call, BrgemmCopyB doesn't know his port descriptors.
    // So we use port descs from source inputs
    const auto element_type = get_input_element_type(0);
    validate_element_type(element_type);
    // The data always store in planar shape after repacking
    const auto planar_pshape = snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_input);
    // data repacking output
    set_output_type(0, element_type, planar_pshape);
    // If compensations are needed, they are provided in 2nd output (which is used in BrgemmCPU)
    if (is_with_compensations()) {
        set_output_type(1, ov::element::f32, planar_pshape);
    }
}

void BrgemmCopyB::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmRepack_validate_and_infer_types);
    const auto& element_type = get_input_element_type(0);
    validate_element_type(element_type);
    const auto port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0));
    const auto shape = ov::Shape(port->get_shape());
    const auto& planar_pshape = snippets::utils::get_planar_pshape(shape, port->get_layout());
    set_output_type(0, element_type, planar_pshape);
    if (is_with_compensations()) {
        set_output_type(1, ov::element::f32, planar_pshape);
    }
}

void BrgemmCopyB::validate_element_type(const ov::element::Type& element_type) {
    OPENVINO_ASSERT(one_of(element_type, element::bf16, element::i8),
                    "BrgemmCopyB doesn't support element type" + element_type.get_type_name());
}

void intel_cpu::BrgemmCopyB::compute_block_size_values(const size_t blk_size_k, const size_t blk_size_n) {
    const auto& input_shape = snippets::utils::get_planar_pshape(input(0)).get_shape();
    m_K_blk = blk_size_k != 0 ? blk_size_k : *(input_shape.rbegin() + 1);
    m_N_blk = blk_size_n != 0 ? blk_size_n : *input_shape.rbegin();
}

ov::Shape intel_cpu::BrgemmCopyB::get_repacking_buffer_shape() const {
    // OneDNN implementation repacks data and always writes the result by m_brgemmVNNIFactor * m_inner_n_block blocks
    // despite the actual size of the input data. Because of that we have to round-up the allocation shape to always have enough memory allocated.
    // For the details, please see 'copy_4x64' and 'copy_2x32' implementations and usage in onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
    return ov::Shape{rnd_up(m_K_blk, m_brgemmVNNIFactor), rnd_up(m_N_blk, m_inner_n_block)};
}

ov::Shape intel_cpu::BrgemmCopyB::get_compensations_buffer_shape() const {
    // Compensations are computed during repacking, so we need to round-up allocation shape according to m_inner_n_block
    // because of OneDNN implementation nuances (as in get_repacking_buffer_shape).
    // However, the compensations are computed by N dimension, so K dimension doesn't affect the compensations buffer
    return ov::Shape{rnd_up(m_N_blk, m_inner_n_block)};
}

std::shared_ptr<Node> intel_cpu::BrgemmCopyB::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmRepack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyB>(new_args.at(0), m_src_type, m_type,
                                         get_input_port_descriptor(0),
                                         get_output_port_descriptor(0),
                                         is_with_compensations() ? get_output_port_descriptor(1) : PortDescriptor{},
                                         snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
                                         m_K_blk, m_N_blk);
}

size_t BrgemmCopyB::get_offset_compensations() const {
    OPENVINO_ASSERT(is_with_compensations() && get_output_size() == 2,
                    "The offset for compensations must be in BrgemmCopyB only with compensations and 2 outputs!");
    return get_output_offset(1);
}

BrgemmCopyB::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& brg_copyb = ov::as_type_ptr<BrgemmCopyB>(n);
    OPENVINO_ASSERT(brg_copyb, "Got invalid node in BrgemmCopyB::ShapeInfer");
    m_layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->input(0))->get_layout();
    m_num_outs = brg_copyb->get_output_size();
}

ov::snippets::IShapeInferSnippets::Result BrgemmCopyB::ShapeInfer::infer(const std::vector<ov::snippets::VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got unexpected number of input shapes");
    const auto planar_shape = ov::snippets::utils::get_planar_vdims(input_shapes[0].get(), m_layout);
    std::vector<ov::snippets::VectorDims> new_shapes(m_num_outs, planar_shape);
    return {new_shapes, ov::snippets::ShapeInferStatus::success};
}

} // namespace intel_cpu
} // namespace ov