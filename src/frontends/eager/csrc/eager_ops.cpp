// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// eager_ops.cpp — Generic OpenVINO-backed ATen op dispatch for eager mode.
//
// Registers a single BOXED FALLBACK at the PrivateUse1 dispatch key that
// intercepts ALL aten ops.  For each op it:
//   1. Checks if the PyTorch frontend has a translator (via get_supported_ops)
//   2. If yes: extracts tensor/scalar args from the stack, builds an
//      EagerGraphDecoder, calls FrontEnd::convert() (the REAL translator),
//      compiles/caches the resulting ov::Model, and infers.
//   3. If no: falls back to CPU via at::native::cpu_fallback.
//
// This means ALL ~400 ops from op_table.cpp work automatically without
// individual wrappers.

#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>

#include "eager_decoder.h"

#include <openvino/frontend/pytorch/frontend.hpp>
#include <openvino/core/model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>
#include <openvino/op/constant.hpp>

#include <cstring>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ov::op;
using namespace ov::frontend::pytorch::eager;

// ═══════════════════════════════════════════════════════════════════════════
// Dtype mapping
// ═══════════════════════════════════════════════════════════════════════════

static ov::element::Type torch_to_ov_dtype(at::ScalarType st) {
    switch (st) {
        case at::kFloat:    return ov::element::f32;
        case at::kDouble:   return ov::element::f64;
        case at::kHalf:     return ov::element::f16;
        case at::kBFloat16: return ov::element::bf16;
        case at::kInt:      return ov::element::i32;
        case at::kLong:     return ov::element::i64;
        case at::kShort:    return ov::element::i16;
        case at::kChar:     return ov::element::i8;
        case at::kByte:     return ov::element::u8;
        case at::kBool:     return ov::element::boolean;
        default:            return ov::element::dynamic;
    }
}

static at::ScalarType ov_to_torch_dtype(ov::element::Type et) {
    if (et == ov::element::f32)      return at::kFloat;
    if (et == ov::element::f64)      return at::kDouble;
    if (et == ov::element::f16)      return at::kHalf;
    if (et == ov::element::bf16)     return at::kBFloat16;
    if (et == ov::element::i32)      return at::kInt;
    if (et == ov::element::i64)      return at::kLong;
    if (et == ov::element::i16)      return at::kShort;
    if (et == ov::element::i8)       return at::kChar;
    if (et == ov::element::u8)       return at::kByte;
    if (et == ov::element::boolean)  return at::kBool;
    return at::kFloat;  // fallback
}

// ═══════════════════════════════════════════════════════════════════════════
// Singletons
// ═══════════════════════════════════════════════════════════════════════════

/// Subclass that skips normalize() (no-op for single-op eager graphs)
/// and exposes get_supported_ops() which is protected in the base class.
class EagerFrontEnd : public ov::frontend::pytorch::FrontEnd {
public:
    void normalize(const std::shared_ptr<ov::Model>&) const override {
        // No-op: skip all ~30 transformation passes for eager mode.
    }

    std::unordered_set<std::string> supported_op_names(
            const ov::frontend::InputModel::Ptr& model) const {
        auto ops = get_supported_ops(model);
        std::unordered_set<std::string> names;
        for (auto& [name, _] : ops)
            names.insert(name);
        return names;
    }
};

static EagerFrontEnd& get_pytorch_fe() {
    static EagerFrontEnd fe;
    return fe;
}

static ov::Core& get_core() {
    static ov::Core core;
    return core;
}

// Set of ops the PyTorch frontend supports (queried once lazily via EagerFrontEnd).
static const std::unordered_set<std::string>& get_supported_op_names() {
    static std::unordered_set<std::string> names = [] {
        auto& fe = get_pytorch_fe();
        // Build a dummy single-node graph to get an InputModel for querying
        std::vector<size_t> dummy_inputs = {1};
        std::vector<size_t> dummy_outputs = {2};
        auto dummy_op = std::make_shared<EagerOpDecoder>(
            "aten::relu", dummy_inputs, dummy_outputs,
            std::vector<ov::Any>{ov::Any(ov::element::f32)},
            std::vector<ov::PartialShape>{ov::PartialShape{1}},
            std::vector<bool>{false});
        auto dummy_graph = std::make_shared<EagerGraphDecoder>(
            dummy_inputs,
            std::vector<ov::Any>{ov::Any(ov::element::f32)},
            std::vector<ov::PartialShape>{ov::PartialShape{1}},
            std::vector<std::shared_ptr<ov::frontend::pytorch::TorchDecoder>>{},
            dummy_op, dummy_outputs);
        dummy_graph->set_input_names({"input_0"});
        auto input_model = fe.load(
            {ov::Any(std::static_pointer_cast<ov::frontend::IDecoder>(dummy_graph))});
        return fe.supported_op_names(input_model);
    }();
    return names;
}

// ═══════════════════════════════════════════════════════════════════════════
// Compiled-model cache + stats
// ═══════════════════════════════════════════════════════════════════════════

struct CacheEntry {
    ov::CompiledModel compiled;
    ov::InferRequest  request;
};

static std::mutex g_cache_mutex;
static std::unordered_map<std::string, CacheEntry> g_cache;
static int64_t g_ov_count = 0;
static double g_convert_us = 0;
static double g_compile_us = 0;

// ═══════════════════════════════════════════════════════════════════════════
// Forward-declare npu_empty from npu_backend.cpp
// ═══════════════════════════════════════════════════════════════════════════

at::Tensor npu_empty(c10::IntArrayRef size, std::optional<at::ScalarType> dtype,
                     std::optional<at::Layout> layout,
                     std::optional<at::Device> device,
                     std::optional<bool> pin_memory,
                     std::optional<at::MemoryFormat> memory_format);

// ═══════════════════════════════════════════════════════════════════════════
// Generic op execution via frontend
// ═══════════════════════════════════════════════════════════════════════════

struct OpInput {
    enum Kind { TENSOR, SCALAR_CONST, NONE };
    Kind kind;
    at::Tensor tensor;                           // for TENSOR
    std::shared_ptr<ov::op::v0::Constant> cst;   // for SCALAR_CONST
};

/// Build cache key from op name + tensor inputs
static std::string make_key(const std::string& op_name,
                             const std::vector<OpInput>& inputs) {
    std::string key = op_name;
    for (auto& inp : inputs) {
        key += "|";
        if (inp.kind == OpInput::TENSOR) {
            key += std::to_string(static_cast<int>(inp.tensor.scalar_type()));
            for (auto d : inp.tensor.sizes())
                key += "," + std::to_string(d);
        } else if (inp.kind == OpInput::SCALAR_CONST) {
            key += "c";
            key += inp.cst->get_element_type().to_string();
        } else {
            key += "N";
        }
    }
    return key;
}

/// Build the OV model via the real frontend for a given op with given inputs.
static std::shared_ptr<ov::Model> build_model_for_op(
        const std::string& aten_op_name,
        const std::vector<OpInput>& inputs) {

    size_t next_id = 1;
    size_t total_op_inputs = inputs.size();

    // Separate into graph params vs constants vs none
    std::vector<size_t> graph_input_ids;
    std::vector<ov::Any> graph_input_types;
    std::vector<ov::PartialShape> graph_input_shapes;
    std::vector<std::string> graph_input_names;

    std::vector<size_t> op_input_ids(total_op_inputs, 0);
    std::vector<ov::Any> op_input_types(total_op_inputs);
    std::vector<ov::PartialShape> op_input_shapes(total_op_inputs);
    std::vector<bool> op_none_mask(total_op_inputs, false);

    std::vector<std::shared_ptr<ov::frontend::pytorch::TorchDecoder>> const_nodes;

    size_t param_idx = 0;
    for (size_t i = 0; i < total_op_inputs; ++i) {
        auto& inp = inputs[i];
        if (inp.kind == OpInput::TENSOR) {
            size_t tid = next_id++;
            auto ov_type = torch_to_ov_dtype(inp.tensor.scalar_type());
            ov::PartialShape shape(std::vector<ov::Dimension>(
                inp.tensor.sizes().begin(), inp.tensor.sizes().end()));

            graph_input_ids.push_back(tid);
            graph_input_types.push_back(ov::Any(ov_type));
            graph_input_shapes.push_back(shape);
            graph_input_names.push_back("input_" + std::to_string(param_idx++));

            op_input_ids[i] = tid;
            op_input_types[i] = ov::Any(ov_type);
            op_input_shapes[i] = shape;
        } else if (inp.kind == OpInput::SCALAR_CONST) {
            size_t tid = next_id++;
            const_nodes.push_back(std::make_shared<EagerConstDecoder>(tid, inp.cst));
            op_input_ids[i] = tid;
            op_input_types[i] = ov::Any(inp.cst->get_element_type());
            op_input_shapes[i] = ov::PartialShape{};
        } else {
            // NONE — use dummy ID
            op_none_mask[i] = true;
            op_input_ids[i] = graph_input_ids.empty() ? next_id : graph_input_ids[0];
        }
    }

    size_t output_tid = next_id++;
    auto op_decoder = std::make_shared<EagerOpDecoder>(
        aten_op_name, op_input_ids, std::vector<size_t>{output_tid},
        op_input_types, op_input_shapes, op_none_mask);

    auto graph_decoder = std::make_shared<EagerGraphDecoder>(
        graph_input_ids, graph_input_types, graph_input_shapes,
        const_nodes, op_decoder, std::vector<size_t>{output_tid});
    graph_decoder->set_input_names(graph_input_names);

    auto& fe = get_pytorch_fe();
    auto input_model = fe.load(
        {ov::Any(std::static_pointer_cast<ov::frontend::IDecoder>(graph_decoder))});
    return fe.convert(input_model);
}

/// Execute via OV: build/cache model, set inputs, infer, return results
static std::vector<at::Tensor> execute_via_ov(
        const std::string& aten_op_name,
        const std::vector<OpInput>& inputs) {

    std::string key = make_key(aten_op_name, inputs);

    CacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_cache.find(key);
        if (it == g_cache.end()) {
            auto _t0 = std::chrono::high_resolution_clock::now();
            auto model = build_model_for_op(aten_op_name, inputs);
            auto _t1 = std::chrono::high_resolution_clock::now();
            auto compiled = get_core().compile_model(model, "CPU");
            auto _t2 = std::chrono::high_resolution_clock::now();
            g_convert_us += std::chrono::duration<double, std::micro>(_t1 - _t0).count();
            g_compile_us += std::chrono::duration<double, std::micro>(_t2 - _t1).count();
            auto request = compiled.create_infer_request();
            auto [ins, ok] = g_cache.emplace(key, CacheEntry{compiled, request});
            entry = &ins->second;
        } else {
            entry = &it->second;
        }
    }

    // Set tensor inputs (only TENSOR kind, in order)
    auto& req = entry->request;
    size_t input_idx = 0;
    for (auto& inp : inputs) {
        if (inp.kind == OpInput::TENSOR) {
            auto ov_tensor = ov::Tensor(
                torch_to_ov_dtype(inp.tensor.scalar_type()),
                ov::Shape(inp.tensor.sizes().begin(), inp.tensor.sizes().end()),
                inp.tensor.data_ptr());
            req.set_input_tensor(input_idx++, ov_tensor);
        }
    }

    req.infer();

    size_t n_out = entry->compiled.outputs().size();
    std::vector<at::Tensor> results;
    results.reserve(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        auto ov_out = req.get_output_tensor(i);
        auto shape = ov_out.get_shape();
        std::vector<int64_t> sizes(shape.begin(), shape.end());
        auto out_dtype = ov_to_torch_dtype(ov_out.get_element_type());
        auto npu_t = npu_empty(sizes, out_dtype,
                                at::Layout::Strided,
                                at::Device(at::DeviceType::PrivateUse1, 0),
                                false, c10::nullopt);
        std::memcpy(npu_t.data_ptr(), ov_out.data(), ov_out.get_byte_size());
        results.push_back(npu_t);
    }

    g_ov_count++;
    return results;
}

// ═══════════════════════════════════════════════════════════════════════════
// Generic boxed fallback
//
// Intercepts ALL ops at PrivateUse1.  For each:
//   - Maps the PyTorch op name to the frontend's aten:: name
//   - If the frontend supports it AND all inputs are tensors/scalars:
//       → route through OV (execute_via_ov)
//   - Otherwise: fall back to CPU
// ═══════════════════════════════════════════════════════════════════════════

/// Convert PyTorch Scalar to OV Constant
static std::shared_ptr<v0::Constant> scalar_to_constant(const at::Scalar& s) {
    if (s.isFloatingPoint()) {
        return v0::Constant::create(ov::element::f64, {}, {s.toDouble()});
    } else if (s.isIntegral(/*includeBool=*/false)) {
        return v0::Constant::create(ov::element::i64, {}, {s.toLong()});
    } else if (s.isBoolean()) {
        return v0::Constant::create(ov::element::boolean, {}, {s.toBool()});
    }
    return nullptr;
}

/// Get the aten:: op name from an OperatorHandle
static std::string get_aten_name(const c10::OperatorHandle& op) {
    auto name = op.operator_name();
    // name.name is like "aten::add", name.overload_name is like "Tensor"
    return std::string(name.name);
}

/// The single generic fallback function
static void npu_generic_fallback(
        const c10::OperatorHandle& op,
        c10::DispatchKeySet dispatch_keys,
        torch::jit::Stack* stack) {

    auto aten_name = get_aten_name(op);

    // Check if the frontend supports this op
    auto& supported = get_supported_op_names();
    if (supported.count(aten_name) == 0) {
        at::native::cpu_fallback(op, stack, /*error_on_views=*/false);
        return;
    }

    // Parse the stack to extract inputs
    const auto& schema = op.schema();
    const auto& args = schema.arguments();
    size_t n_args = args.size();
    auto stack_start = stack->size() - n_args;

    // Check if we can handle all args (tensors, scalars, None)
    // and if at least one tensor is on NPU
    bool has_npu_tensor = false;
    bool can_handle = true;
    std::vector<OpInput> op_inputs;
    op_inputs.reserve(n_args);

    // Track .out variant output tensors (args with writable alias like Tensor(a!))
    std::vector<at::Tensor> out_tensors;

    for (size_t i = 0; i < n_args; ++i) {
        const auto& iv = (*stack)[stack_start + i];

        // Check if this argument is a writable output (e.g. Tensor(a!) out)
        const auto* alias = args[i].alias_info();
        if (alias && alias->isWrite()) {
            if (iv.isTensor() && iv.toTensor().defined()) {
                out_tensors.push_back(iv.toTensor());
            }
            continue;  // Don't add to op_inputs
        }

        if (iv.isTensor()) {
            auto& t = iv.toTensor();
            if (!t.defined()) {
                op_inputs.push_back({OpInput::NONE, {}, nullptr});
            } else {
                if (t.device().type() == at::DeviceType::PrivateUse1)
                    has_npu_tensor = true;
                // Ensure tensor is contiguous for zero-copy
                op_inputs.push_back({OpInput::TENSOR, t.contiguous(), nullptr});
            }
        } else if (iv.isScalar()) {
            auto cst = scalar_to_constant(iv.toScalar());
            if (cst)
                op_inputs.push_back({OpInput::SCALAR_CONST, {}, cst});
            else
                can_handle = false;
        } else if (iv.isNone()) {
            op_inputs.push_back({OpInput::NONE, {}, nullptr});
        } else if (iv.isBool()) {
            auto cst = v0::Constant::create(ov::element::boolean, {}, {iv.toBool()});
            op_inputs.push_back({OpInput::SCALAR_CONST, {}, cst});
        } else if (iv.isInt()) {
            auto cst = v0::Constant::create(ov::element::i64, {}, {iv.toInt()});
            op_inputs.push_back({OpInput::SCALAR_CONST, {}, cst});
        } else if (iv.isDouble()) {
            auto cst = v0::Constant::create(ov::element::f64, {}, {iv.toDouble()});
            op_inputs.push_back({OpInput::SCALAR_CONST, {}, cst});
        } else {
            // Complex types (lists, strings, etc.) — can't handle generically
            can_handle = false;
            break;
        }
    }

    if (!can_handle || !has_npu_tensor) {
        at::native::cpu_fallback(op, stack, /*error_on_views=*/false);
        return;
    }

    // Try to execute through OV
    try {
        auto results = execute_via_ov(aten_name, op_inputs);

        // Pop inputs from stack and push outputs
        torch::jit::drop(*stack, n_args);

        // For .out variants: copy result into the pre-allocated out tensor(s)
        if (!out_tensors.empty() && !results.empty()) {
            for (size_t i = 0; i < std::min(out_tensors.size(), results.size()); ++i) {
                auto& out_t = out_tensors[i];
                auto& res_t = results[i];
                if (out_t.sizes() != res_t.sizes())
                    out_t.resize_(res_t.sizes());
                std::memcpy(out_t.data_ptr(), res_t.data_ptr(), res_t.nbytes());
            }
            for (auto& t : out_tensors)
                stack->push_back(std::move(t));
        } else {
            const auto& returns = schema.returns();
            if (results.size() == 1 && returns.size() == 1) {
                stack->push_back(std::move(results[0]));
            } else {
                for (auto& r : results)
                    stack->push_back(std::move(r));
            }
        }
    } catch (const std::exception& e) {
        // OV conversion/execution failed, fall back to CPU
        at::native::cpu_fallback(op, stack, /*error_on_views=*/false);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Registration — single boxed fallback catches ALL ops at PrivateUse1
// ═══════════════════════════════════════════════════════════════════════════

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_generic_fallback>());
}

// ═══════════════════════════════════════════════════════════════════════════
// Python bindings for stats
// ═══════════════════════════════════════════════════════════════════════════

void register_eager_ops_bindings(py::module& m) {
    m.def("ov_op_count", []() { return g_ov_count; },
          "Number of ops executed through OpenVINO");
    m.def("reset_ov_stats", []() { g_ov_count = 0; g_convert_us = 0; g_compile_us = 0; });
    m.def("ov_convert_us", []() { return g_convert_us; });
    m.def("ov_compile_us", []() { return g_compile_us; });
}
