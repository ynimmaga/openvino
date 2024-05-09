// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "dummy_node.hpp"
#include "graph.h"
#include "nodes/input.h"
#include "nodes/reorder.h"
#include "nodes/transpose.h"

#include "openvino/op/transpose.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"

#include "common_test_utils/node_builders/constant.hpp"

using namespace ov::intel_cpu;
using LOOK = Edge::LOOK;

struct Result {
    size_t transpose_count;
    size_t optimized_reorder_count;
    size_t not_optimized_reorder_count;
};

struct MergeTransposeReorderTestParam {
    LayoutType firstNodeLayout;
    LOOK firstNodeInplaceDirection;
    LayoutType lastNodeLayout;
    LOOK lastNodeInplaceDirection;
    size_t num_consumers;
    Result test_result;
};

using MergeTransposeReorderTestParams = std::tuple<ov::Shape, MergeTransposeReorderTestParam>;

class MergeTransposeReorderCPUTest : public testing::WithParamInterface<MergeTransposeReorderTestParams>,
                                     public ov::test::TestsCommon {
public:
    void Validate() const {
        const auto& result = std::get<1>(GetParam()).test_result;
        CheckTransposeCount(result.transpose_count);
        CheckReorderCount(result.optimized_reorder_count, result.not_optimized_reorder_count);
    }

protected:
    void SetUp() override {
        const auto& shape = std::get<0>(GetParam());
        const auto& params = std::get<1>(GetParam());
        CreateGraph(shape,
                    params.firstNodeLayout,
                    params.firstNodeInplaceDirection,
                    params.lastNodeLayout,
                    params.lastNodeInplaceDirection,
                    params.num_consumers);
    }

    /* graph topology
        ┌───────┐
        │ Input │
        └───┬───┘
            │
        ┌───┴───┐
        │ Dummy │      <*NOTE: fake node with firstNodeLayout, and firstNodeInplaceDirection*>
        └───┬───┘
            │
       ┌────┴────┐
       │Transpose│     <*NOTE: Reorder is inserted before/after Transpose depending on first/second node layouts.*>
       └────┬────┘
            │
        ┌───┴───┐
        │ Dummy │      <*NOTE: fake node with lastNodeLayout, and lastNodeInplaceDirection*>
        └───┬───┘
            │
       ┌────┴───┐
       │ Output │
       └────────┘
    */
    void CreateGraph(const ov::Shape& testShape,
                     LayoutType firstNodeLayout,
                     LOOK firstNodeInplaceDirection,
                     LayoutType lastNodeLayout,
                     LOOK lastNodeInplaceDirection,
                     size_t num_consumers) {
        Config conf;
        conf.rtCacheCapacity = 100;
        auto context = std::make_shared<GraphContext>(conf, nullptr, false);
        const dnnl::engine cpuEngine = context->getEngine();
        m_graph = std::unique_ptr<Graph>(new Graph());

        const auto precision = ov::element::f32;
        OPENVINO_ASSERT(testShape.size() == 4 || testShape.size() == 3, "Only 4D and 3D shapes are supported");
        // ov::Model with only a transpose node
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(precision, testShape)};
        auto order = testShape.size() == 4 ? std::vector<int32_t>{0, 3, 1, 2} : std::vector<int32_t>{0, 2, 1};
        auto constOrder = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{order.size()}, order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(params[0], constOrder);
        ov::ResultVector results;
        for (size_t i = 0; i < num_consumers; i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(transpose));

        // Replicate
        auto replicate = [&](std::vector<NodePtr> &nodes, std::vector<EdgePtr> &edges) -> void {
            std::unordered_set<NodePtr> nodesSet;

            auto addEdge = [&](const NodePtr& parent, const NodePtr& child, size_t parentPort, size_t childPort) -> void {
                auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
                Node::addEdge(edge);
                edges.push_back(edge);
                nodesSet.insert(parent);
                nodesSet.insert(child);
            };

            auto inputNode = std::make_shared<node::Input>(params[0], context);

            auto dummyNode1 = std::make_shared<cpu_unit_test::DummyNode>(
                testShape, precision, "reshape", "DummyNode", context, firstNodeLayout, firstNodeInplaceDirection);

            auto orderNode = std::make_shared<node::Input>(constOrder, context);
            auto transposeNode = std::make_shared<node::Transpose>(transpose, context);
            transposeNode->filterSupportedPrimitiveDescriptors();

            addEdge(inputNode, dummyNode1, 0, 0);
            addEdge(dummyNode1, transposeNode, 0, 0);
            addEdge(orderNode, transposeNode, 0, 1);

            const auto& transpose_shape = transpose->get_output_shape(0);
            for (size_t i = 0; i < num_consumers; i++) {
                auto dummyConsumer = std::make_shared<cpu_unit_test::DummyNode>(transpose_shape,
                                                                                precision,
                                                                                "multiply",
                                                                                "DummyNode",
                                                                                context,
                                                                                lastNodeLayout,
                                                                                lastNodeInplaceDirection);
                auto outputNode = std::make_shared<node::Input>(results[i], context);
                addEdge(transposeNode, dummyConsumer, 0, 0);
                addEdge(dummyConsumer, outputNode, 0, 0);
            }

            for (auto &node : nodesSet) nodes.emplace_back(node);
        };

        std::vector<NodePtr> graphNodes;
        std::vector<EdgePtr> graphEdges;
        replicate(graphNodes, graphEdges);

        m_graph->CreateGraph(graphNodes, graphEdges, context, "fused_graph");
    }

    void CheckTransposeCount(size_t ref_transpose_count) const {
        size_t transpose_count = 0;
        for (auto &node : m_graph->GetNodes()) {
            if (node->getType() == Type::Transpose) {
                transpose_count++;
            }
        }
        ASSERT_EQ(ref_transpose_count, transpose_count);
    }

    void CheckReorderCount(size_t ref_optimized_reorder_count, size_t ref_not_optimized_reorder_count) const {
        size_t optimized_count = 0;
        size_t not_optimized_count = 0;
        for (auto &node : m_graph->GetNodes()) {
            if (auto reorder_node = std::dynamic_pointer_cast<node::Reorder>(node)) {
                if (reorder_node->getOptimized())
                    optimized_count++;
                else
                    not_optimized_count++;
            }
        }
        ASSERT_EQ(ref_optimized_reorder_count, optimized_count);
        ASSERT_EQ(ref_not_optimized_reorder_count, not_optimized_count);
    }

private:
    std::unique_ptr<Graph> m_graph;
};  // class MergeTransposeReorderCPUTest

TEST_P(MergeTransposeReorderCPUTest, smoke_Run_MergeTransposeReorder) {
    Validate();
}

const std::vector<ov::Shape> input_shapes{{1, 3, 8, 16}, {3, 8, 16}};

const std::vector<MergeTransposeReorderTestParam> test_params = {
    // upstream node or downstream node is inPlaced thereby the inserted Reorder cannot be optimized.
    {LayoutType::ncsp, LOOK::LOOK_UP, LayoutType::nspc, LOOK::LOOK_DOWN, 1, Result{0, 0, 2}},
    // no inplace conflict: a single optimized reorder fused with Transpose
    {LayoutType::ncsp, LOOK::LOOK_DOWN, LayoutType::nspc, LOOK::LOOK_UP, 1, Result{0, 1, 1}},
    // 3 non-inplace consumers share a single optimized reorder fused with Transpose
    {LayoutType::ncsp, LOOK::LOOK_UP, LayoutType::nspc, LOOK::LOOK_UP, 3, Result{0, 1, 3}},
    // 3 inplace consumers cannot share reorders thus transpose is not fused with reorders
    // there will be also 3 reorders between 3 dummyNode-consumers and 3 Result nodes
    {LayoutType::ncsp, LOOK::LOOK_UP, LayoutType::nspc, LOOK::LOOK_DOWN, 3, Result{1, 0, 6}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Run_MergeTransposeReorder,
                         MergeTransposeReorderCPUTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::ValuesIn(test_params)));
