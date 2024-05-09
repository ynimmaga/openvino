// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/loop.hpp"

#include <gtest/gtest.h>

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/cleanup_loop_offsets.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/lowered/pass/optimize_loop_single_evaluation.hpp"
#include "snippets/lowered/pass/validate_unified_loops.hpp"
#include "snippets/lowered/pass/validate_expanded_loops.hpp"
#include "snippets/lowered/pass/normalize_loop_ids.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "subgraph_simple.hpp"

using Snippets_TailProcessingTransformation = ::testing::Test;
// [Inserted Loop number, [ptr_increments, final_offsets]
using ref_map = std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>>;
using namespace ov::snippets::lowered;

constexpr static size_t vector_size = 16;

static void init_linear_ir(const std::vector<ov::PartialShape>& in_shapes, LinearIR& linear_ir, size_t block_size) {
    auto body = ov::test::snippets::AddFunction(in_shapes).getOriginal();
    auto shape_infer_factory = std::make_shared<ov::snippets::IShapeInferSnippetsFactory>();
    linear_ir = LinearIR(body, shape_infer_factory);
    auto expr_it = std::find_if(linear_ir.cbegin(), linear_ir.cend(),
                                [](const ExpressionPtr& expr) { return ov::is_type<ov::op::v1::Add>(expr->get_node()); });
    ASSERT_TRUE(expr_it != linear_ir.cend());
    const auto add = *expr_it;
    const auto loop_input_ports = std::vector<ExpressionPort>{add->get_input_port(0), add->get_input_port(1)};
    const auto loop_output_ports = std::vector<ExpressionPort>{add->get_output_port(0)};
    const auto loop_manager = linear_ir.get_loop_manager();
    const auto in_shape0 = in_shapes[0].get_shape();
    const auto in_shape1 = in_shapes[1].get_shape();
    const auto inner_wa = std::max(*in_shape0.rbegin(), *in_shape1.rbegin());
    const auto inner_inc = std::min(vector_size, inner_wa);
    const auto blocked_wa = block_size;
    const auto blocked_inc = 1;
    const auto outer_wa = std::max(*(in_shape0.rbegin() + 1), *(in_shape1.rbegin() + 1));
    const auto outer_inc = blocked_wa;
    loop_manager->mark_loop(expr_it, std::next(expr_it), inner_wa, inner_inc, 0, loop_input_ports, loop_output_ports);
    loop_manager->mark_loop(expr_it, std::next(expr_it), blocked_wa, blocked_inc, 1, loop_input_ports, loop_output_ports);
    const auto loop_id = loop_manager->mark_loop(expr_it, std::next(expr_it), outer_wa, outer_inc, 1, loop_input_ports, loop_output_ports);
    const auto& outer_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
    const auto outer_tail_size = outer_wa % outer_inc;
    if (outer_tail_size != 0) {
        outer_loop_info->register_pass_to_handler<SpecificLoopIterType::LAST_ITER, pass::TransformInnerSplitLoop>(outer_tail_size);
    }
}

static void apply_transformations(LinearIR& linear_ir, const std::shared_ptr<pass::PassConfig>& config) {
    const auto is_loop_decomp_disabled = config->is_disabled<pass::InsertSpecificIterations>();
    if (is_loop_decomp_disabled) {
        config->disable<pass::ValidateExpandedLoops>();
    }

    pass::PassPipeline pipeline(config);
    pipeline.register_pass<pass::InsertLoadStore>(vector_size);
    pipeline.register_pass<pass::ValidateUnifiedLoops>();
    pipeline.register_pass<pass::InitLoops>();
    pipeline.register_pass<pass::InsertLoops>();
    pipeline.register_pass<pass::InsertSpecificIterations>();
    pipeline.register_pass<pass::NormalizeLoopIDs>(!is_loop_decomp_disabled);
    pipeline.register_pass<pass::ValidateExpandedLoops>();
    pipeline.register_pass<pass::CleanupLoopOffsets>();
    pipeline.register_pass<pass::OptimizeLoopSingleEvaluation>();
    pipeline.run(linear_ir);
}

static void validate(const LinearIR& linear_ir, const ref_map& reference) {
    size_t loop_num = 0;
    for (const auto& expr : linear_ir) {
        const auto& node = expr->get_node();
        ASSERT_TRUE(!ov::is_type<ov::snippets::op::LoopBeginDynamic>(node) && !ov::is_type<ov::snippets::op::LoopEndDynamic>(node));
        const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEndStatic>(node);
        if (!loop_end)
            continue;
        ASSERT_GT(reference.count(loop_num), 0);
        ASSERT_TRUE(loop_end->get_ptr_increments() == reference.at(loop_num).first);
        ASSERT_TRUE(loop_end->get_finalization_offsets() == reference.at(loop_num).second);
        loop_num++;
    }
    ASSERT_EQ(loop_num, reference.size());
}

TEST(Snippets_TailProcessingTransformation, BlockedWOTail_OriginalPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {1, 2, 16, 20};
    ov::Shape inputShape1 = {1, 2, 16, 20};
    init_linear_ir({inputShape0, inputShape1}, linear_ir, 4);

    auto config = std::make_shared<pass::PassConfig>();
    config->disable<pass::CleanupLoopOffsets>();
    config->disable<pass::InsertSpecificIterations>();
    config->disable<pass::OptimizeLoopSingleEvaluation>();
    apply_transformations(linear_ir, config);

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 1), std::vector<int64_t>(3, -20)};
    reference[1] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, -80)};
    reference[2] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, -320)};

    validate(linear_ir, reference);
}

TEST(Snippets_TailProcessingTransformation, BlockedWOTail_CleanUpPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {1, 2, 16, 20};
    ov::Shape inputShape1 = {1, 2, 16, 20};
    init_linear_ir({inputShape0, inputShape1}, linear_ir, 4);

    auto config = std::make_shared<pass::PassConfig>();
    config->disable<pass::InsertSpecificIterations>();
    config->disable<pass::OptimizeLoopSingleEvaluation>();
    apply_transformations(linear_ir, config);

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 1), std::vector<int64_t>(3, 0)};
    reference[1] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 0)};
    reference[2] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 0)};

    validate(linear_ir, reference);
}

TEST(Snippets_TailProcessingTransformation, BlockedTail_OriginalPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {1, 2, 18, 20};
    ov::Shape inputShape1 = {1, 2, 18, 20};
    init_linear_ir({inputShape0, inputShape1}, linear_ir, 4);

    auto config = std::make_shared<pass::PassConfig>();
    config->disable<pass::CleanupLoopOffsets>();
    apply_transformations(linear_ir, config);

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 16)};  // Vector Inner
    reference[1] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, -16)};  // Blocked Inner
    reference[2] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, -80)};  // Vector Blocked
    reference[3] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, 0)}; // Vector Outer

    reference[4] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 16)};  // Vector Inner
    reference[5] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, -16)};  // Blocked Inner
    reference[6] = { std::vector<int64_t>(3, 20), std::vector<int64_t>(3, -40)};  // Tail Blocked
    reference[7] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, -320)};  // Tail Blocked

    validate(linear_ir, reference);
}

TEST(Snippets_TailProcessingTransformation, BlockedTail_CleanUpPtrShifts) {
    LinearIR linear_ir;
    ov::Shape inputShape0 = {1, 2, 18, 20};
    ov::Shape inputShape1 = {1, 2, 18, 20};
    init_linear_ir({inputShape0, inputShape1}, linear_ir, 4);

    apply_transformations(linear_ir, std::make_shared<pass::PassConfig>());

    // [Inserted Loop number, [ptr_increments, final_offsets]
    std::map<size_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> reference;
    reference[0] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 16)};  // Vector Inner
    reference[1] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 4)};  // Blocked Inner
    reference[2] = {std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 0)};   // Vector Blocked
    reference[3] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 0)}; // Vector Outer

    reference[4] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 16)};  // Vector Inner
    reference[5] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 4)};  // Blocked Inner
    reference[6] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 0)};  // Tail Blocked
    reference[7] = { std::vector<int64_t>(3, 0), std::vector<int64_t>(3, 0)};  // Tail Blocked

    validate(linear_ir, reference);
}