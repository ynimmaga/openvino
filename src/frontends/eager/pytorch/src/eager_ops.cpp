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
#include <ATen/ExpandUtils.h>

#include "eager_decoder.h"

#include <openvino/frontend/pytorch/frontend.hpp>
#include <openvino/core/model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/unsqueeze.hpp>

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

// Per-op last-compile info — for introspection of "what OV ops were created".
static std::mutex g_last_mutex;
static std::string                       g_last_aten_name;
static std::vector<std::string>          g_last_ov_ops;       // OV op type_info names
static std::unordered_map<std::string, std::vector<std::string>> g_per_op_ov_ops;

// Per-op execution counters: how many calls were routed via OV vs CPU fallback.
static std::unordered_map<std::string, int64_t> g_per_op_ov_calls;
static std::unordered_map<std::string, int64_t> g_per_op_cpu_calls;

// ═══════════════════════════════════════════════════════════════════════════
// Generic op execution via frontend
// ═══════════════════════════════════════════════════════════════════════════

/// Forward declarations
at::Tensor npu_empty(c10::IntArrayRef size, std::optional<at::ScalarType> dtype,
                     std::optional<at::Layout> layout,
                     std::optional<at::Device> device,
                     std::optional<bool> pin_memory,
                     std::optional<at::MemoryFormat> memory_format);
at::Tensor npu_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                              std::optional<at::ScalarType> dtype,
                              std::optional<at::Layout> layout,
                              std::optional<at::Device> device,
                              std::optional<bool> pin_memory);

/// Make a contiguous NPU copy of a (possibly non-contiguous) NPU tensor WITHOUT
/// going through the dispatcher.  Calling `t.contiguous()` directly would
/// dispatch `aten::contiguous` → decomposed into `aten::clone` → re-enter our
/// boxed fallback for the SAME tensor → infinite recursion → stack/heap
/// corruption.  Since NPU storage is plain CPU memory we manually walk the
/// strides and copy element-by-element into a fresh contiguous NPU tensor.
/// Make a contiguous NPU copy of a (possibly non-contiguous) NPU tensor WITHOUT
/// going through the dispatcher.  Calling `t.contiguous()` directly would
/// dispatch `aten::contiguous` → decomposed into `aten::clone` → re-enter our
/// boxed fallback for the SAME tensor → infinite recursion → stack/heap
/// corruption.  Since NPU storage is plain CPU memory we manually walk the
/// strides and copy element-by-element into a fresh contiguous NPU tensor.
static at::Tensor make_contiguous_npu(const at::Tensor& t) {
    if (!t.defined() || t.numel() == 0 || t.is_contiguous()) {
        return t;
    }
    // Allocate a fresh contiguous NPU tensor.
    auto npu_t = npu_empty(t.sizes(), t.scalar_type(),
                            at::Layout::Strided,
                            at::Device(at::DeviceType::PrivateUse1, 0),
                            false, c10::nullopt);
    // Copy element-by-element walking the source strides.
    const auto itemsize = static_cast<int64_t>(t.dtype().itemsize());
    const auto* src_base = static_cast<const char*>(t.storage().data()) +
                           static_cast<int64_t>(t.storage_offset()) * itemsize;
    auto* dst = static_cast<char*>(npu_t.data_ptr());

    const auto sizes   = t.sizes();
    const auto strides = t.strides();
    const int ndim = static_cast<int>(sizes.size());

    if (ndim == 0) {
        std::memcpy(dst, src_base, itemsize);
        return npu_t;
    }

    std::vector<int64_t> idx(ndim, 0);
    int64_t total = t.numel();
    for (int64_t k = 0; k < total; ++k) {
        int64_t off_elems = 0;
        for (int d = 0; d < ndim; ++d) off_elems += idx[d] * strides[d];
        std::memcpy(dst + k * itemsize, src_base + off_elems * itemsize, itemsize);
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < sizes[d]) break;
            idx[d] = 0;
        }
    }
    return npu_t;
}

struct OpInput {
    enum Kind { TENSOR, SCALAR_CONST, NONE, TENSOR_LIST };
    Kind kind;
    at::Tensor tensor;                           // for TENSOR
    std::shared_ptr<ov::op::v0::Constant> cst;   // for SCALAR_CONST (also IntList/BoolList/DoubleList baked as Constant)
    std::vector<at::Tensor> tensor_list;         // for TENSOR_LIST
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
            // Bake the constant value into the key so different scalar
            // values (e.g. different `dim` for cat/mean) get distinct
            // compiled graphs.
            const auto& byts = inp.cst->get_byte_size();
            const char* p = static_cast<const char*>(inp.cst->get_data_ptr());
            for (size_t i = 0; i < byts; ++i) {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%02x", (unsigned char)p[i]);
                key += buf;
            }
        } else if (inp.kind == OpInput::TENSOR_LIST) {
            key += "L" + std::to_string(inp.tensor_list.size());
            for (auto& t : inp.tensor_list) {
                key += ":";
                key += std::to_string(static_cast<int>(t.scalar_type()));
                for (auto d : t.sizes())
                    key += "," + std::to_string(d);
            }
        } else {
            key += "N";
        }
    }
    return key;
}

// ─── Custom OV builders for ops with list-typed args ────────────────────────
// These bypass the PT frontend (which expects prim::ListConstruct nodes for
// list inputs that our boxed dispatcher can't produce).  Each builder creates
// an ov::Model with one Parameter per tensor input (in stable order) and one
// Result.  The boxed `execute_via_ov` then binds inputs / reads the output
// like for any other compiled op.
//
// Returns nullptr if the op isn't handled here.
static std::shared_ptr<ov::Model> try_build_custom_model(
        const std::string& aten_op_name,
        const std::vector<OpInput>& inputs) {

    auto make_param = [](const at::Tensor& t, const std::string& name) {
        auto et = torch_to_ov_dtype(t.scalar_type());
        ov::PartialShape ps(std::vector<ov::Dimension>(
            t.sizes().begin(), t.sizes().end()));
        auto p = std::make_shared<v0::Parameter>(et, ps);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        return p;
    };

    // ── aten::cat(tensors, dim=0) ─────────────────────────────────────────
    if (aten_op_name == "aten::cat" || aten_op_name == "aten::concat" ||
        aten_op_name == "aten::concatenate") {
        if (inputs.size() < 1 || inputs[0].kind != OpInput::TENSOR_LIST)
            return nullptr;
        int64_t dim = 0;
        if (inputs.size() >= 2 && inputs[1].kind == OpInput::SCALAR_CONST) {
            // Constant holds a single i64
            auto vals = inputs[1].cst->cast_vector<int64_t>();
            if (!vals.empty()) dim = vals[0];
        }
        ov::ParameterVector params;
        ov::OutputVector cat_inputs;
        size_t k = 0;
        for (auto& t : inputs[0].tensor_list) {
            auto p = make_param(t, "input_" + std::to_string(k++));
            params.push_back(p);
            cat_inputs.push_back(p);
        }
        auto cat = std::make_shared<v0::Concat>(cat_inputs, dim);
        auto result = std::make_shared<v0::Result>(cat);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, params, "cat_eager");
    }

    // ── aten::mean.dim(self, dim, keepdim=False, *, dtype=None) ───────────
    // ── aten::sum.dim_IntList(self, dim, keepdim=False, *, dtype=None) ────
    if ((aten_op_name == "aten::mean" || aten_op_name == "aten::sum") &&
        inputs.size() >= 2 &&
        inputs[0].kind == OpInput::TENSOR &&
        inputs[1].kind == OpInput::SCALAR_CONST) {

        bool keepdim = false;
        if (inputs.size() >= 3 && inputs[2].kind == OpInput::SCALAR_CONST) {
            auto v = inputs[2].cst->cast_vector<int8_t>();
            if (!v.empty()) keepdim = (v[0] != 0);
        }
        auto p = make_param(inputs[0].tensor, "input_0");
        auto axes = inputs[1].cst;  // already an i64 constant or vector i64
        std::shared_ptr<ov::Node> reduce_node;
        if (aten_op_name == "aten::mean") {
            reduce_node = std::make_shared<v1::ReduceMean>(p, axes, keepdim);
        } else {
            reduce_node = std::make_shared<v1::ReduceSum>(p, axes, keepdim);
        }
        auto result = std::make_shared<v0::Result>(reduce_node);
        return std::make_shared<ov::Model>(
            ov::ResultVector{result}, ov::ParameterVector{p},
            aten_op_name + "_eager");
    }

    // ── aten::stack(tensors, dim=0) ───────────────────────────────────────
    // Implemented as Concat(Unsqueeze(t, dim) for t in tensors).
    if (aten_op_name == "aten::stack") {
        if (inputs.size() < 1 || inputs[0].kind != OpInput::TENSOR_LIST)
            return nullptr;
        int64_t dim = 0;
        if (inputs.size() >= 2 && inputs[1].kind == OpInput::SCALAR_CONST) {
            auto vals = inputs[1].cst->cast_vector<int64_t>();
            if (!vals.empty()) dim = vals[0];
        }
        auto axis_c = v0::Constant::create(ov::element::i64, {1}, {dim});
        ov::ParameterVector params;
        ov::OutputVector cat_inputs;
        size_t k = 0;
        for (auto& t : inputs[0].tensor_list) {
            auto p = make_param(t, "input_" + std::to_string(k++));
            params.push_back(p);
            cat_inputs.push_back(std::make_shared<v0::Unsqueeze>(p, axis_c));
        }
        auto cat = std::make_shared<v0::Concat>(cat_inputs, dim);
        auto result = std::make_shared<v0::Result>(cat);
        return std::make_shared<ov::Model>(
            ov::ResultVector{result}, params, "stack_eager");
    }

    return nullptr;
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
            // Try custom builder first (handles list-typed args like cat/mean/stack)
            auto model = try_build_custom_model(aten_op_name, inputs);
            if (!model) {
                model = build_model_for_op(aten_op_name, inputs);
            }
            auto _t1 = std::chrono::high_resolution_clock::now();
            auto compiled = get_core().compile_model(model, "CPU");
            auto _t2 = std::chrono::high_resolution_clock::now();
            g_convert_us += std::chrono::duration<double, std::micro>(_t1 - _t0).count();
            g_compile_us += std::chrono::duration<double, std::micro>(_t2 - _t1).count();
            // Record the OV op types in the freshly built model for introspection
            {
                std::vector<std::string> op_types;
                for (const auto& node : model->get_ordered_ops()) {
                    op_types.emplace_back(node->get_type_name());
                }
                std::lock_guard<std::mutex> lk(g_last_mutex);
                g_last_aten_name = aten_op_name;
                g_last_ov_ops    = op_types;
                g_per_op_ov_ops[aten_op_name] = op_types;
            }
            auto request = compiled.create_infer_request();
            auto [ins, ok] = g_cache.emplace(key, CacheEntry{compiled, request});
            entry = &ins->second;
        } else {
            entry = &it->second;
        }
    }

    // Set tensor inputs by writing into the InferRequest's OV-owned input
    // buffers.  This avoids passing raw torch data pointers to OV, which the
    // InferRequest keeps a reference to past the end of this call and which
    // can become dangling (causing heap corruption at shutdown) when the
    // input was a temporary contiguous copy of a non-contiguous torch tensor.
    auto& req = entry->request;

    // Helper: copy a torch tensor into the next OV input slot, handling
    // both contiguous and non-contiguous (stride-walked) layouts.
    size_t input_idx = 0;
    auto bind_tensor = [&](const at::Tensor& t) {
        auto ov_in = req.get_input_tensor(input_idx++);
        const auto nbytes = ov_in.get_byte_size();
        if (nbytes == 0) return;

        const auto itemsize = static_cast<int64_t>(t.dtype().itemsize());
        const auto* src_base = static_cast<const char*>(t.storage().data()) +
                               static_cast<int64_t>(t.storage_offset()) * itemsize;
        auto* dst = static_cast<char*>(ov_in.data());

        if (t.is_contiguous()) {
            std::memcpy(dst, src_base, nbytes);
        } else {
            const auto sizes   = t.sizes();
            const auto strides = t.strides();
            const int ndim = static_cast<int>(sizes.size());
            std::vector<int64_t> idx(ndim, 0);
            int64_t total = t.numel();
            for (int64_t k = 0; k < total; ++k) {
                int64_t off_elems = 0;
                for (int d = 0; d < ndim; ++d) off_elems += idx[d] * strides[d];
                std::memcpy(dst + k * itemsize,
                            src_base + off_elems * itemsize, itemsize);
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++idx[d] < sizes[d]) break;
                    idx[d] = 0;
                }
            }
        }
    };

    for (auto& inp : inputs) {
        if (inp.kind == OpInput::TENSOR) {
            bind_tensor(inp.tensor);
        } else if (inp.kind == OpInput::TENSOR_LIST) {
            for (auto& t : inp.tensor_list) bind_tensor(t);
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

        // PyTorch's dispatcher will frequently pre-allocate or restride the
        // boxed-fallback output to match the strides of the FIRST tensor input
        // (this is the standard "stride-propagating" convention used for
        // pointwise ops).  If we hand back a contiguously-allocated output and
        // the dispatcher then `as_strided`s it onto a smaller storage layout,
        // any subsequent CPU copy walks past the end of our buffer and
        // corrupts the heap.
        //
        // To stay safe, when the OV output shape matches the first tensor
        // input's shape and that input is non-contiguous, allocate the result
        // using the input's strides (via empty_strided) and stride-copy the
        // OV (contiguous) result into it.
        const at::Tensor* ref_input = nullptr;
        for (auto& inp : inputs) {
            if (inp.kind == OpInput::TENSOR && inp.tensor.defined()) {
                if (inp.tensor.sizes().size() == sizes.size()) {
                    bool same = true;
                    for (size_t d = 0; d < sizes.size(); ++d)
                        if (inp.tensor.sizes()[d] != sizes[d]) { same = false; break; }
                    if (same) { ref_input = &inp.tensor; break; }
                }
            }
        }

        // Decide output stride policy:
        //   - "Materializing" ops (clone / contiguous / _to_copy / copy_)
        //     produce a freshly contiguous output regardless of input layout.
        //     Allocate contig and write the OV result contiguously.
        //   - All other ops (pointwise like neg/add/mul/...) follow the
        //     PyTorch stride-propagation policy where output strides are the
        //     *dense permutation* of the first input's stride order
        //     (`at::infer_dense_strides`).  If we returned a different
        //     layout, the dispatcher's subsequent restride reads garbage.
        static const std::unordered_set<std::string> kMaterializeOps = {
            "aten::clone", "aten::contiguous", "aten::_to_copy",
            "aten::copy_", "aten::lift_fresh", "aten::lift_fresh_copy"
        };
        bool use_input_layout = ref_input && !ref_input->is_contiguous() &&
                                kMaterializeOps.count(aten_op_name) == 0;

        at::Tensor npu_t;
        if (use_input_layout) {
            // PyTorch's pointwise-op stride-propagation policy is NOT to copy
            // the input's strides verbatim — it builds a *dense* output
            // whose dimensions are permuted to match the order implied by
            // the input's strides (so the output is contiguous in that
            // permuted order).  Using the raw input strides leaves holes in
            // storage and the dispatcher's subsequent restride reads
            // garbage.  `at::infer_dense_strides` returns the correct
            // permuted-dense strides for a given (sizes, input_strides).
            auto out_strides = at::infer_dense_strides(
                ref_input->sizes(), ref_input->strides());
            npu_t = npu_empty_strided(sizes, out_strides, out_dtype,
                                       at::Layout::Strided,
                                       at::Device(at::DeviceType::PrivateUse1, 0),
                                       false);
            // Stride-walk copy from contig OV result to dense-but-permuted dest
            const auto itemsize = static_cast<int64_t>(c10::elementSize(out_dtype));
            const char* src = static_cast<const char*>(ov_out.data());
            char* dst_base = static_cast<char*>(npu_t.data_ptr());
            const auto strides = npu_t.strides();
            const int ndim = static_cast<int>(sizes.size());
            std::vector<int64_t> idx(ndim, 0);
            int64_t total = npu_t.numel();
            for (int64_t k = 0; k < total; ++k) {
                int64_t off_elems = 0;
                for (int d = 0; d < ndim; ++d) off_elems += idx[d] * strides[d];
                std::memcpy(dst_base + off_elems * itemsize, src + k * itemsize, itemsize);
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++idx[d] < sizes[d]) break;
                    idx[d] = 0;
                }
            }
        } else {
            npu_t = npu_empty(sizes, out_dtype,
                                at::Layout::Strided,
                                at::Device(at::DeviceType::PrivateUse1, 0),
                                false, c10::nullopt);
            std::memcpy(npu_t.data_ptr(), ov_out.data(), ov_out.get_byte_size());
        }
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

/// Canonicalize aten op names whose only difference from a translator-backed
/// op is a couple of "noise" args.  Returns the canonical name plus a list of
/// input-arg indices to drop (mask out) before handing to the frontend.
///
/// Note: `aten::masked_fill_` is handled specially in the dispatcher because
/// it writes back to its first argument and isn't a pure functional rename.
struct CanonResult {
    std::string canonical;       // empty if no rewrite
    std::vector<size_t> drop;    // arg indices to mask out (none-ify) before FE
};
static CanonResult canonicalize_op(const std::string& name) {
    // aten::_softmax(self, dim, half_to_float)  ->  aten::softmax(self, dim, None=dtype)
    // (drop arg 2; translator only reads dim and optional dtype)
    if (name == "aten::_softmax")        return {"aten::softmax",         {2}};
    if (name == "aten::_safe_softmax")   return {"aten::softmax",         {2}};
    // aten::_reshape_alias(self, size, stride)  ->  aten::reshape(self, size)
    if (name == "aten::_reshape_alias")  return {"aten::reshape",         {2}};
    // aten::_log_softmax(self, dim, half_to_float)  ->  aten::log_softmax(self, dim, None)
    if (name == "aten::_log_softmax")    return {"aten::log_softmax",     {2}};
    return {};
}

/// The single generic fallback function
static void npu_generic_fallback(
        const c10::OperatorHandle& op,
        c10::DispatchKeySet dispatch_keys,
        torch::jit::Stack* stack) {

    auto raw_aten_name = get_aten_name(op);
    auto canon = canonicalize_op(raw_aten_name);
    const std::string& aten_name = canon.canonical.empty() ? raw_aten_name : canon.canonical;

    // Check if the frontend supports this op
    auto& supported = get_supported_op_names();
    if (supported.count(aten_name) == 0) {
        {
            std::lock_guard<std::mutex> lk(g_last_mutex);
            g_per_op_cpu_calls[raw_aten_name]++;
        }
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
                // Don't materialize a contiguous copy here — execute_via_ov()
                // walks the strides directly when copying into the OV-owned
                // input buffer.  That avoids creating temporary NPU tensors
                // (which trigger aten::clone → re-entry into this fallback).
                op_inputs.push_back({OpInput::TENSOR, t, nullptr});
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
        } else if (iv.isIntList()) {
            // Bake list of ints as an i64 Constant of shape [N].
            auto vec = iv.toIntList().vec();
            auto cst = v0::Constant::create(ov::element::i64,
                                            {vec.size()}, vec);
            op_inputs.push_back({OpInput::SCALAR_CONST, {}, cst});
        } else if (iv.isBoolList()) {
            auto lst = iv.toBoolList();
            std::vector<char> vec;
            vec.reserve(lst.size());
            for (auto b : lst) vec.push_back(b ? 1 : 0);
            auto cst = v0::Constant::create(ov::element::boolean,
                                            {vec.size()}, vec);
            op_inputs.push_back({OpInput::SCALAR_CONST, {}, cst});
        } else if (iv.isDoubleList()) {
            auto vec = iv.toDoubleList().vec();
            auto cst = v0::Constant::create(ov::element::f64,
                                            {vec.size()}, vec);
            op_inputs.push_back({OpInput::SCALAR_CONST, {}, cst});
        } else if (iv.isTensorList()) {
            std::vector<at::Tensor> tlist;
            auto lst = iv.toTensorList();
            tlist.reserve(lst.size());
            for (size_t li = 0; li < lst.size(); ++li) {
                at::Tensor t = lst.get(li);
                if (!t.defined()) { can_handle = false; break; }
                if (t.device().type() == at::DeviceType::PrivateUse1)
                    has_npu_tensor = true;
                tlist.push_back(std::move(t));
            }
            if (!can_handle) break;
            op_inputs.push_back({OpInput::TENSOR_LIST, {}, nullptr, std::move(tlist)});
        } else {
            // Complex types (strings, dicts, etc.) — can't handle generically
            can_handle = false;
            break;
        }
    }

    // Apply canonicalization's drop mask: nullify args the canonical op
    // doesn't expect (e.g. `_softmax`'s `half_to_float` bool).
    for (auto idx : canon.drop) {
        if (idx < op_inputs.size()) {
            op_inputs[idx] = {OpInput::NONE, {}, nullptr};
        }
    }

    if (!can_handle || !has_npu_tensor) {
        {
            std::lock_guard<std::mutex> lk(g_last_mutex);
            g_per_op_cpu_calls[raw_aten_name]++;
        }
        at::native::cpu_fallback(op, stack, /*error_on_views=*/false);
        return;
    }

    // Try to execute through OV
    try {
        auto results = execute_via_ov(aten_name, op_inputs);
        {
            std::lock_guard<std::mutex> lk(g_last_mutex);
            g_per_op_ov_calls[raw_aten_name]++;
        }

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
        {
            std::lock_guard<std::mutex> lk(g_last_mutex);
            g_per_op_cpu_calls[raw_aten_name]++;
        }
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
    m.def("reset_ov_stats", []() {
        g_ov_count = 0; g_convert_us = 0; g_compile_us = 0;
        std::lock_guard<std::mutex> lk(g_last_mutex);
        g_last_aten_name.clear();
        g_last_ov_ops.clear();
        g_per_op_ov_ops.clear();
        g_per_op_ov_calls.clear();
        g_per_op_cpu_calls.clear();
    });
    m.def("ov_convert_us", []() { return g_convert_us; });
    m.def("ov_compile_us", []() { return g_compile_us; });
    m.def("supported_op_names", []() {
        const auto& s = get_supported_op_names();
        return std::vector<std::string>(s.begin(), s.end());
    }, "Aten op names the OpenVINO PyTorch frontend can translate.");
    m.def("last_ov_ops", []() {
        std::lock_guard<std::mutex> lk(g_last_mutex);
        return std::make_pair(g_last_aten_name, g_last_ov_ops);
    }, "(aten_name, [ov_op_type, ...]) for the most recently compiled op.");
    m.def("per_op_ov_ops", []() {
        std::lock_guard<std::mutex> lk(g_last_mutex);
        return g_per_op_ov_ops;
    }, "Map: aten op name -> list of OV op type names produced by translator.");
    m.def("per_op_exec_counts", []() {
        std::lock_guard<std::mutex> lk(g_last_mutex);
        std::unordered_map<std::string, std::pair<int64_t,int64_t>> out;
        for (auto& kv : g_per_op_ov_calls)  out[kv.first].first  = kv.second;
        for (auto& kv : g_per_op_cpu_calls) out[kv.first].second = kv.second;
        return out;
    }, "Map: aten op name -> (ov_calls, cpu_fallback_calls).");
}
