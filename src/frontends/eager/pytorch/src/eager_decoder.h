// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// eager_decoder.h — Lightweight TorchDecoder implementations for eager mode.
//
// Provides two decoder classes that represent a single-op "graph":
//
//   EagerGraphDecoder  — the graph-level decoder (parameters + one op + outputs)
//   EagerOpDecoder     — the op-node decoder (aten::add, aten::relu, etc.)
//
// Together they allow calling FrontEnd::load(decoder) + FrontEnd::convert()
// which internally invokes the REAL translator functions from the frontend
// (translate_add, translate_mul, translate_1to1_match, etc.).

#pragma once

#include <openvino/frontend/pytorch/decoder.hpp>
#include <openvino/op/constant.hpp>

#include <string>
#include <vector>

namespace ov {
namespace frontend {
namespace pytorch {
namespace eager {

// ═══════════════════════════════════════════════════════════════════════════
// EagerOpDecoder — represents a single ATen op node inside the graph.
//
// The frontend's convert_pytorch_model iterates nodes via visit_subgraph.
// For each node it creates a NodeContext using this decoder, then looks up
// the translator (e.g. translate_add) by op_type and calls it.
// ═══════════════════════════════════════════════════════════════════════════

class EagerOpDecoder : public TorchDecoder {
public:
    /// \param op_type       Full op type, e.g. "aten::add"
    /// \param input_ids     Tensor IDs this node consumes (from tensor_map)
    /// \param output_ids    Tensor IDs this node produces
    /// \param input_types   Element type of each input (ov::Any wrapping ov::element::Type)
    /// \param input_shapes  Shape of each input
    /// \param none_mask     Which inputs are "None" (absent/default)
    EagerOpDecoder(std::string op_type,
                   std::vector<size_t> input_ids,
                   std::vector<size_t> output_ids,
                   std::vector<ov::Any> input_types,
                   std::vector<ov::PartialShape> input_shapes,
                   std::vector<bool> none_mask = {})
        : m_op_type(std::move(op_type)),
          m_inputs(std::move(input_ids)),
          m_outputs(std::move(output_ids)),
          m_input_types(std::move(input_types)),
          m_input_shapes(std::move(input_shapes)),
          m_none_mask(std::move(none_mask)) {
        m_none_mask.resize(m_inputs.size(), false);
        // Build debug name strings from tensor IDs (must be numeric)
        for (auto id : m_inputs)
            m_input_debug_names.push_back(std::to_string(id));
        for (auto id : m_outputs)
            m_output_debug_names.push_back(std::to_string(id));
    }

    // --- Core identification ---
    const std::string& get_op_type() const override { return m_op_type; }
    const std::string& get_schema() const override { return m_empty; }

    // --- Inputs ---
    const std::vector<size_t>& inputs() const override { return m_inputs; }
    size_t get_input_size() const { return m_inputs.size(); }
    bool input_is_none(size_t i) const override { return i < m_none_mask.size() && m_none_mask[i]; }
    ov::Any get_input_type(size_t i) const override { return m_input_types.at(i); }
    ov::PartialShape get_input_shape(size_t i) const override { return m_input_shapes.at(i); }
    const std::vector<size_t>& get_input_strides(size_t) const override { return m_empty_strides; }
    const std::string& get_input_debug_name(size_t i) const override { return m_input_debug_names.at(i); }
    const std::string& get_input_signature_name(size_t i) const override { return m_input_debug_names.at(i); }

    // --- Outputs ---
    const std::vector<size_t>& outputs() const override { return m_outputs; }
    size_t output(size_t i) const override { return m_outputs.at(i); }
    size_t num_of_outputs() const override { return m_outputs.size(); }
    size_t output_list_size() const override { return 0; }
    ov::PartialShape get_output_shape(size_t) const override { return ov::PartialShape::dynamic(); }
    ov::Any get_output_type(size_t) const override { return ov::Any(ov::element::dynamic); }
    const std::string& get_output_debug_name(size_t i) const override { return m_output_debug_names.at(i); }

    // --- Node annotation ---
    std::shared_ptr<ov::Node> mark_node(std::shared_ptr<ov::Node> n) const override { return n; }
    bool may_produce_alias(size_t, size_t) const override { return false; }

    // --- Not needed for eager ---
    ov::Any const_input(size_t) const override { return {}; }
    ov::OutputVector try_decode_get_attr() const override { return {}; }
    ov::OutputVector as_constant() const override { return {}; }
    const std::string& as_string() const override { return m_empty; }
    size_t get_subgraph_size() const override { return 0; }
    void visit_subgraph(std::function<void(std::shared_ptr<TorchDecoder>)>) const override {}
    std::shared_ptr<TorchDecoder> get_subgraph_decoder(size_t) const override { return nullptr; }
    bool is_input_inlined(size_t) const override { return false; }
    std::shared_ptr<TorchDecoder> get_inlined_input_decoder(size_t) const override { return nullptr; }
    ov::Any get_attribute(const std::string&) const override { return {}; }
    size_t get_named_input(const std::string&) const override { return 0; }
    const std::string& decoder_type_name() const override { return m_decoder_type; }
    DecoderRTInfo get_rt_info() const override { return {}; }
    bool has_converter() const override { return false; }
    ov::OutputVector convert(const ov::frontend::NodeContext*) const override { return {}; }

private:
    std::string m_op_type;
    std::vector<size_t> m_inputs;
    std::vector<size_t> m_outputs;
    std::vector<ov::Any> m_input_types;
    std::vector<ov::PartialShape> m_input_shapes;
    std::vector<bool> m_none_mask;
    std::string m_empty;
    std::string m_decoder_type = "ts";
    std::vector<size_t> m_empty_strides;
    std::vector<std::string> m_input_debug_names;
    std::vector<std::string> m_output_debug_names;
};

// ═══════════════════════════════════════════════════════════════════════════
// EagerConstDecoder — represents a prim::Constant node for scalar args.
//
// Used for non-default alpha in add/sub, or similar constant inputs.
// ═══════════════════════════════════════════════════════════════════════════

class EagerConstDecoder : public TorchDecoder {
public:
    EagerConstDecoder(size_t output_id, std::shared_ptr<ov::op::v0::Constant> constant)
        : m_output_id(output_id),
          m_outputs({output_id}),
          m_constant(std::move(constant)),
          m_output_debug_name(std::to_string(output_id)) {}

    const std::string& get_op_type() const override { return m_op_type; }
    const std::string& get_schema() const override { return m_empty; }
    ov::OutputVector as_constant() const override { return {m_constant}; }

    // Inputs: constants have no inputs
    const std::vector<size_t>& inputs() const override { return m_empty_vec; }
    bool input_is_none(size_t) const override { return true; }
    ov::Any get_input_type(size_t) const override { return {}; }
    ov::PartialShape get_input_shape(size_t) const override { return {}; }
    const std::vector<size_t>& get_input_strides(size_t) const override { return m_empty_vec; }
    const std::string& get_input_debug_name(size_t) const override { return m_empty; }
    const std::string& get_input_signature_name(size_t) const override { return m_empty; }

    // Outputs: one constant output
    const std::vector<size_t>& outputs() const override { return m_outputs; }
    size_t output(size_t) const override { return m_output_id; }
    size_t num_of_outputs() const override { return 1; }
    size_t output_list_size() const override { return 0; }
    ov::PartialShape get_output_shape(size_t) const override { return ov::PartialShape{}; }
    ov::Any get_output_type(size_t) const override {
        return ov::Any(m_constant->get_element_type());
    }
    const std::string& get_output_debug_name(size_t) const override { return m_output_debug_name; }

    std::shared_ptr<ov::Node> mark_node(std::shared_ptr<ov::Node> n) const override { return n; }
    bool may_produce_alias(size_t, size_t) const override { return false; }

    ov::Any const_input(size_t) const override { return {}; }
    ov::OutputVector try_decode_get_attr() const override { return {}; }
    const std::string& as_string() const override { return m_empty; }
    size_t get_subgraph_size() const override { return 0; }
    void visit_subgraph(std::function<void(std::shared_ptr<TorchDecoder>)>) const override {}
    std::shared_ptr<TorchDecoder> get_subgraph_decoder(size_t) const override { return nullptr; }
    bool is_input_inlined(size_t) const override { return false; }
    std::shared_ptr<TorchDecoder> get_inlined_input_decoder(size_t) const override { return nullptr; }
    ov::Any get_attribute(const std::string&) const override { return {}; }
    size_t get_named_input(const std::string&) const override { return 0; }
    const std::string& decoder_type_name() const override { return m_decoder_type; }
    DecoderRTInfo get_rt_info() const override { return {}; }
    bool has_converter() const override { return false; }
    ov::OutputVector convert(const ov::frontend::NodeContext*) const override { return {}; }

private:
    std::string m_op_type = "prim::Constant";
    std::string m_empty;
    std::string m_decoder_type = "ts";
    size_t m_output_id;
    std::vector<size_t> m_outputs;
    std::vector<size_t> m_empty_vec;
    std::shared_ptr<ov::op::v0::Constant> m_constant;
    std::string m_output_debug_name;
};

// ═══════════════════════════════════════════════════════════════════════════
// EagerGraphDecoder — represents the entire single-op "graph".
//
// The FrontEnd's convert() calls:
//   1. inputs()            → get graph input tensor IDs
//   2. get_input_shape(i)  → get shape for each parameter
//   3. get_input_type(i)   → get dtype for each parameter
//   4. visit_subgraph(cb)  → iterate nodes: constants first, then the op
//   5. outputs() / output(i) / num_of_outputs() → graph output tensor IDs
// ═══════════════════════════════════════════════════════════════════════════

class EagerGraphDecoder : public TorchDecoder {
public:
    /// \param graph_inputs   Tensor IDs for graph parameters
    /// \param input_types    OV element types for each parameter
    /// \param input_shapes   Shapes for each parameter
    /// \param const_nodes    Constant decoder nodes to visit before the op
    /// \param op_node        The single op node decoder
    /// \param graph_outputs  Tensor IDs for graph results
    EagerGraphDecoder(std::vector<size_t> graph_inputs,
                      std::vector<ov::Any> input_types,
                      std::vector<ov::PartialShape> input_shapes,
                      std::vector<std::shared_ptr<TorchDecoder>> const_nodes,
                      std::shared_ptr<TorchDecoder> op_node,
                      std::vector<size_t> graph_outputs)
        : m_inputs(std::move(graph_inputs)),
          m_input_types(std::move(input_types)),
          m_input_shapes(std::move(input_shapes)),
          m_const_nodes(std::move(const_nodes)),
          m_op_node(std::move(op_node)),
          m_outputs(std::move(graph_outputs)) {}

    // --- Graph inputs ---
    const std::vector<size_t>& inputs() const override { return m_inputs; }
    ov::Any get_input_type(size_t i) const override { return m_input_types.at(i); }
    ov::PartialShape get_input_shape(size_t i) const override { return m_input_shapes.at(i); }
    const std::vector<size_t>& get_input_strides(size_t) const override { return m_empty_vec; }
    const std::string& get_input_debug_name(size_t i) const override {
        return i < m_input_debug_names.size() ? m_input_debug_names.at(i) : m_empty;
    }
    const std::string& get_input_signature_name(size_t i) const override { return m_input_names.at(i); }
    bool input_is_none(size_t) const override { return false; }

    // --- Graph outputs ---
    const std::vector<size_t>& outputs() const override { return m_outputs; }
    size_t output(size_t i) const override { return m_outputs.at(i); }
    size_t num_of_outputs() const override { return m_outputs.size(); }
    size_t output_list_size() const override { return 0; }
    ov::PartialShape get_output_shape(size_t) const override { return ov::PartialShape::dynamic(); }
    ov::Any get_output_type(size_t) const override { return ov::Any(ov::element::dynamic); }
    const std::string& get_output_debug_name(size_t i) const override {
        return i < m_output_debug_names.size() ? m_output_debug_names.at(i) : m_empty;
    }

    // --- Subgraph (the op body) ---
    size_t get_subgraph_size() const override { return 1; }
    void visit_subgraph(std::function<void(std::shared_ptr<TorchDecoder>)> visitor) const override {
        for (auto& c : m_const_nodes) visitor(c);
        visitor(m_op_node);
    }
    std::shared_ptr<TorchDecoder> get_subgraph_decoder(size_t) const override { return nullptr; }

    // --- Identification ---
    const std::string& get_op_type() const override { return m_graph_type; }
    const std::string& get_schema() const override { return m_empty; }
    std::shared_ptr<ov::Node> mark_node(std::shared_ptr<ov::Node> n) const override { return n; }
    bool may_produce_alias(size_t, size_t) const override { return false; }

    // --- Not needed for graph decoder ---
    ov::Any const_input(size_t) const override { return {}; }
    ov::OutputVector try_decode_get_attr() const override { return {}; }
    ov::OutputVector as_constant() const override { return {}; }
    const std::string& as_string() const override { return m_empty; }
    bool is_input_inlined(size_t) const override { return false; }
    std::shared_ptr<TorchDecoder> get_inlined_input_decoder(size_t) const override { return nullptr; }
    ov::Any get_attribute(const std::string&) const override { return {}; }
    size_t get_named_input(const std::string&) const override { return 0; }
    const std::string& decoder_type_name() const override { return m_decoder_type; }
    DecoderRTInfo get_rt_info() const override { return {}; }
    bool has_converter() const override { return false; }
    ov::OutputVector convert(const ov::frontend::NodeContext*) const override { return {}; }

    /// Must be called after construction to set input names
    void set_input_names(std::vector<std::string> names) {
        m_input_names = std::move(names);
        // Build numeric debug names from tensor IDs
        m_input_debug_names.clear();
        for (auto id : m_inputs)
            m_input_debug_names.push_back(std::to_string(id));
        m_output_debug_names.clear();
        for (auto id : m_outputs)
            m_output_debug_names.push_back(std::to_string(id));
    }

private:
    std::vector<size_t> m_inputs;
    std::vector<ov::Any> m_input_types;
    std::vector<ov::PartialShape> m_input_shapes;
    std::vector<std::shared_ptr<TorchDecoder>> m_const_nodes;
    std::shared_ptr<TorchDecoder> m_op_node;
    std::vector<size_t> m_outputs;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_input_debug_names;
    std::vector<std::string> m_output_debug_names;
    std::string m_graph_type = "graph";
    std::string m_empty;
    std::string m_decoder_type = "ts";
    std::vector<size_t> m_empty_vec;
};

}  // namespace eager
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
