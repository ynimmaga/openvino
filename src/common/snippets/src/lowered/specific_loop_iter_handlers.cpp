// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/specific_loop_iter_handlers.hpp"

#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/lowered/pass/propagate_subtensors.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

SpecificIterationHandlers::SpecificIterationHandlers(size_t loop_work_amount, size_t loop_increment) {
    const auto tail_size = utils::is_dynamic_value(loop_work_amount) ? 1lu : loop_work_amount % loop_increment;
    if (tail_size != 0) {
        m_last_iter_handlers.register_pass<lowered::pass::UpdateMemoryAccessCounts>(tail_size);
        m_last_iter_handlers.register_pass<lowered::pass::UpdateSubtensors>(tail_size);
    }
}

SpecificIterationHandlers::SpecificIterationHandlers(lowered::pass::PassPipeline first_iter_handlers,
                                                     lowered::pass::PassPipeline main_body_handlers,
                                                     lowered::pass::PassPipeline last_iter_handlers)
    : m_first_iter_handlers(std::move(first_iter_handlers)),
      m_main_body_handlers(std::move(main_body_handlers)),
      m_last_iter_handlers(std::move(last_iter_handlers)) {}

SpecificIterationHandlers SpecificIterationHandlers::merge_handlers(
    const SpecificIterationHandlers& lhs,
    const SpecificIterationHandlers& rhs) {
    return SpecificIterationHandlers(
        pass::PassPipeline::merge_pipelines(lhs.m_first_iter_handlers, rhs.m_first_iter_handlers),
        pass::PassPipeline::merge_pipelines(lhs.m_main_body_handlers, rhs.m_main_body_handlers),
        pass::PassPipeline::merge_pipelines(lhs.m_last_iter_handlers, rhs.m_last_iter_handlers));
}

} // namespace lowered
} // namespace snippets
} // namespace ov
