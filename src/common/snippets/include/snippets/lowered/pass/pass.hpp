// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass_config.hpp"
#include "snippets/pass/positioned_pass.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface Pass
 * @brief Base class for transformations on linear IR
 * @ingroup snippets
 */
class Pass {
public:
    Pass() = default;
    virtual ~Pass() = default;
    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static {"Pass"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }

    /**
     * @brief Apply the pass to the Linear IR
     * @param linear_ir the target Linear IR
     * @return status of the pass
     */
    virtual bool run(lowered::LinearIR& linear_ir) = 0;
};

class PassPipeline {
public:
    using PositionedPassLowered = snippets::pass::PositionedPass<lowered::pass::Pass>;

    PassPipeline();
    PassPipeline(const std::shared_ptr<PassConfig>& pass_config);

    void register_pass(const snippets::pass::PassPosition& position, const std::shared_ptr<Pass>& pass);
    void register_pass(const std::shared_ptr<Pass>& pass);

    template<typename T, class... Args>
    void register_pass(Args&&... args) {
        static_assert(std::is_base_of<Pass, T>::value, "Pass not derived from lowered::Pass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        register_pass(pass);
    }
    template<typename T, class Pos, class... Args, std::enable_if<std::is_same<snippets::pass::PassPosition, Pos>::value, bool>() = true>
    void register_pass(const snippets::pass::PassPosition& position, Args&&... args) {
        static_assert(std::is_base_of<Pass, T>::value, "Pass not derived from lowered::Pass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        register_pass(position, pass);
    }

    void register_positioned_passes(const std::vector<PositionedPassLowered>& pos_passes);

    void run(lowered::LinearIR& linear_ir) const;

private:
    std::shared_ptr<PassConfig> m_pass_config;
    std::vector<std::shared_ptr<Pass>> m_passes;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
