// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_mha.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string MHA::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj) {
    std::vector<ov::PartialShape> inputShapes;
    std::vector<ov::element::Type> elem_types;
    ov::element::Type prc;
    bool withMul;
    size_t thread_count;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    ov::AnyMap additionalConfig;
    std::tie(inputShapes,
             elem_types,
             prc,
             withMul,
             thread_count,
             num_nodes,
             num_subgraphs,
             targetDevice,
             additionalConfig) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i)
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({inputShapes[i]}) << "_";
    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i << "]=" << elem_types[i] << "_";
    result << "Mul=" << withMul << "_";
    result << "ThreadNum=" << thread_count << "_";
    result << "PRC=" << prc << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice << "_";

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }
    return result.str();
}

void MHA::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    ov::element::Type prc;
    ov::AnyMap additionalConfig;
    std::tie(inputShapes,
             m_input_types,
             prc,
             m_with_mul,
             m_thread_count,
             ref_num_nodes,
             ref_num_subgraphs,
             targetDevice,
             additionalConfig) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    const auto subgraph_model = get_subgraph();
    function = subgraph_model->getOriginal();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }

    setInferenceType(prc);
    inType = outType = prc;
    if (prc == ov::element::bf16)
        rel_threshold = 0.05f;
}

void MHA::compile_model() {
    if (m_thread_count != default_thread_count)
        core->set_property(targetDevice, ov::inference_num_threads(m_thread_count));
    SubgraphBaseTest::compile_model();
}

void MHA::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& model_inputs = function->inputs();
    for (int i = 0; i < model_inputs.size(); ++i) {
        const auto& model_input = model_inputs[i];
        ov::Tensor tensor;
        ov::test::utils::InputGenerateData in_data;
        // To avoid big relative errors in the vicinity of zero, only positive values are generated for bf16 precision
        in_data.start_from = model_input.get_element_type() == ov::element::bf16 ? 0 : -1;
        in_data.range = 2;
        in_data.resolution = 256;
        tensor =
            ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), model_input.get_shape(), in_data);
        inputs.insert({model_input.get_node_shared_ptr(), tensor});
    }
}

std::shared_ptr<SnippetsFunctionBase> MHA::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAFunction>(inputDynamicShapes, m_input_types, m_with_mul);
}

void MHASelect::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto model_inputs = function->inputs();
    for (auto& model_input : model_inputs) {
        const auto node_input = model_input.get_node_shared_ptr();
        const auto name = node_input->get_friendly_name();
        ov::Tensor tensor;
        int seed = 0;
        if (name.find("less") != std::string::npos) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -2;
            in_data.range = 5 + seed;
            in_data.resolution = 10;
            in_data.seed = seed;
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                                                             model_input.get_shape(),
                                                             in_data);
            seed++;
        } else {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -1;
            in_data.range = 2;
            in_data.resolution = 256;
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                                                             model_input.get_shape(),
                                                             in_data);
        }
        inputs.insert({node_input, tensor});
    }
}

std::shared_ptr<SnippetsFunctionBase> MHASelect::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHASelectFunction>(inputDynamicShapes, m_input_types);
}

std::shared_ptr<SnippetsFunctionBase> MHAWOTransposeOnInputs::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAWOTransposeOnInputsFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAWOTranspose::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAWOTransposeFunction>(inputDynamicShapes, m_input_types);
}

std::shared_ptr<SnippetsFunctionBase> MHAINT8MatMul::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAINT8MatMulFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAQuantMatMul0::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAQuantMatMul0Function>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAFQAfterMatMul::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAFQAfterMatMulFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAFQ::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAFQFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAMulAdd::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAMulAddFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHATransposedB::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHATransposedInputFunction>(inputDynamicShapes, true);
}

std::shared_ptr<SnippetsFunctionBase> MHAWithExtractedReshape::get_subgraph() {
    return std::make_shared<ov::test::snippets::MHAWithExtractedReshapeFunction>(inputDynamicShapes, false);
}

TEST_P(MHA, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHASelect, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWOTransposeOnInputs, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWOTranspose, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAMulAdd, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHATransposedB, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAINT8MatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAQuantMatMul0, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAFQAfterMatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAFQ, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWithExtractedReshape, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
