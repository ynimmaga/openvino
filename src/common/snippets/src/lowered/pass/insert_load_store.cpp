// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

InsertLoadStore::InsertLoadStore(size_t vector_size) : m_vector_size(vector_size) {}

size_t InsertLoadStore::get_count(const ExpressionPort& port) const {
    const auto layout = port.get_descriptor_ptr()->get_layout();
    const auto shape = port.get_descriptor_ptr()->get_shape();
    size_t last_dim_idx = 0;
    if (port.get_type() == ExpressionPort::Type::Input)
        last_dim_idx = utils::get_input_dim_idx(layout, 0);
    else if (port.get_type() == ExpressionPort::Type::Output)
        last_dim_idx = utils::get_output_dim_idx(layout, 0);
    else
        OPENVINO_THROW("Unsupported type of expression port");
    const auto dim = shape[last_dim_idx];
    return utils::is_dynamic_value(dim) ? m_vector_size : std::min(dim, m_vector_size);
}

bool InsertLoadStore::insert_load(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it) {
    const auto& shape_infer_seq = utils::get_first_child_shape_infer_expr_seq(*data_expr_it);
    const std::shared_ptr<Expression>& data_expr = shape_infer_seq.empty() ? *data_expr_it : shape_infer_seq.back();
    const auto& data_ngraph_output = data_expr->get_node()->output(0);
    bool was_inserted = false;
    const auto& data_out = data_expr->get_output_port_connector(0);
    for (const auto& consumer_input : data_out->get_consumers()) {
        const auto& consumer_expr = consumer_input.get_expr();
        const auto& consumer = consumer_expr->get_node();
        const auto ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(consumer);
        if (ma && ma->is_memory_access_input_port(consumer_input.get_index()))
            return false;

        const auto load = std::make_shared<op::Load>(data_ngraph_output, get_count(data_expr->get_output_port(0)));
        linear_ir.insert_node(load, std::vector<PortConnectorPtr>{ data_out }, consumer_expr->get_loop_ids(),
                              true, linear_ir.find_after(data_expr_it, consumer_expr), { consumer_input });
        was_inserted = true;
    }

    return was_inserted;
}

bool InsertLoadStore::insert_store(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it) {
    const auto& shape_infer_seq = utils::get_first_parent_shape_infer_expr_seq(*data_expr_it);
    const auto& data_expr = shape_infer_seq.empty() ? *data_expr_it : shape_infer_seq.back();

    const auto& parent_output = data_expr->get_input_port_connector(0)->get_source();
    const auto& parent_expr = parent_output.get_expr();
    const auto port = parent_output.get_index();
    const auto& parent = parent_expr->get_node();
    const auto ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(parent);
    if (ma && ma->is_memory_access_output_port(port))
        return false;

    const auto& loop_ids = parent_expr->get_loop_ids();
    const auto store = std::make_shared<op::Store>(parent->output(port), get_count(data_expr->get_input_port(0)));
    const auto& insertion_pos = linear_ir.find_after(std::reverse_iterator<LinearIR::constExprIt>(data_expr_it), parent_expr).base();
    linear_ir.insert_node(store, std::vector<ExpressionPort>{ parent_output }, loop_ids, true, insertion_pos, { data_expr->get_input_port(0) });
    return true;
}

bool InsertLoadStore::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertLoadStore")

    bool modified = false;
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<ov::op::v0::Parameter>(node)) {
            modified |= insert_load(linear_ir, expr_it);
        } else if (ov::is_type<ov::op::v0::Result>(node)) {
            modified |= insert_store(linear_ir, expr_it);
        } else if (ov::is_type<op::Buffer>(node)) {
            modified |= insert_load(linear_ir, expr_it);
            if (ov::is_type<op::IntermediateMemoryBuffer>(node))
                modified |= insert_store(linear_ir, expr_it);
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
