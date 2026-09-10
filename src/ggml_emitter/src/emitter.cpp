// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The GraphEmitter subclass that turns the GGUF decoder builder's add_op() callbacks into ggml
// tensors, plus the backend/buffer plumbing needed to execute the result.
//
// OpenVINO's GGUF frontend links no ggml: it only calls virtual methods on GraphEmitter. This
// file is the only place in the tree where ggml headers appear.

#include "openvino/ggml_emitter/emitter.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include "builder/gguf_builder.hpp"
#include "builder/graph_emitter.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace ggml_emitter {

using namespace ov::frontend::gguf;

namespace {

constexpr int MAX_NODES = 8192;

// ov::PartialShape is [ne3,ne2,ne1,ne0]; ggml's ne[] runs the other way.
std::vector<int64_t> to_ne(const ov::PartialShape& ps, int64_t dyn) {
    std::vector<int64_t> ne(4, 1);
    const size_t r = ps.size();
    for (size_t i = 0; i < r && i < 4; i++) {
        const auto& d = ps[r - 1 - i];
        ne[i] = d.is_static() ? d.get_length() : dyn;
    }
    return ne;
}

// Caller-owned: the emitter is destroyed with the DecoderBuilder when the build call returns,
// so everything the caller needs afterwards lives here.
struct Out {
    std::map<std::string, ggml_tensor*> map, externals;
    std::set<std::string> unsupported;
    ggml_tensor* last = nullptr;
};

class GgmlEmitter : public GraphEmitter {
public:
    GgmlEmitter(std::unordered_map<std::string, ov::Tensor>& weights,
                std::unordered_map<std::string, GgufTensorType>& qtypes,
                std::string arch,
                ggml_context* ctx,
                ggml_context* ext_ctx,
                std::map<std::string, ggml_tensor*>* weights_by_name,
                int n_tokens,
                int n_kv,
                int rope_mode,
                Out* out)
        : GraphEmitter(weights, qtypes, std::move(arch)),
          m_ctx(ctx),
          m_ext_ctx(ext_ctx),
          m_w(weights_by_name),
          m_n_tokens(n_tokens),
          m_n_kv(n_kv),
          m_rope_mode(rope_mode),
          m_out(out) {}

    std::string add_op(const std::string& op_type,
                       const std::string& name,
                       const std::vector<std::string>& inputs,
                       const ov::PartialShape& out_shape,
                       ov::element::Type out_type,
                       int op_case,
                       std::map<std::string, ov::Any> attrs) override {
        // blocks/ query shape_of_tensor(), so keep the base's bookkeeping alive.
        record_tensor_meta(name, out_shape, out_type);

        ggml_tensor* t = build(op_type, name, inputs, out_shape, op_case, attrs);
        if (t) {
            ggml_set_name(t, name.c_str());
            m_out->map[name] = t;
            m_out->last = t;
        } else {
            m_out->unsupported.insert(op_type);
        }
        return name;
    }

    std::shared_ptr<ov::op::v0::Parameter> add_input(const std::string& name,
                                                     ov::element::Type type,
                                                     const ov::PartialShape& shape) override {
        auto p = GraphEmitter::add_input(name, type, shape);
        const bool cache = name.rfind("cache_", 0) == 0;
        auto ne = to_ne(shape, cache ? m_n_kv : m_n_tokens);
        ggml_type gt = GGML_TYPE_F32;
        if (cache) {
            gt = GGML_TYPE_F16;
        } else if (name == "inp_tokens" || name == "inp_pos" || name == "inp_out_ids") {
            gt = GGML_TYPE_I32;
        } else if (name == "inp_kv_idx") {
            gt = GGML_TYPE_I64;
        } else if (name.rfind("self_kq_mask", 0) == 0) {
            gt = GGML_TYPE_F16;
            ne[0] = m_n_kv;
            ne[1] = GGML_PAD(m_n_tokens, 64);  // llama.cpp pads the mask rows to 64
        }
        ggml_tensor* t = ggml_new_tensor_4d(m_ext_ctx, gt, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(t, name.c_str());
        if (!cache) {
            ggml_set_input(t);
        }
        m_out->map[name] = t;
        m_out->externals[name] = t;
        return p;
    }

private:
    ggml_tensor* get(const std::vector<std::string>& in, size_t i) {
        if (i >= in.size()) {
            return nullptr;
        }
        auto it = m_out->map.find(in[i]);
        if (it != m_out->map.end()) {
            return it->second;
        }
        auto w = m_w->find(in[i]);
        return w == m_w->end() ? nullptr : w->second;
    }

    ggml_tensor* build(const std::string& op,
                       const std::string& name,
                       const std::vector<std::string>& in,
                       const ov::PartialShape& out_shape,
                       int op_case,
                       std::map<std::string, ov::Any>& a) {
        auto attr = [&](const char* k) -> ov::Any {
            auto it = a.find(k);
            return it == a.end() ? ov::Any() : it->second;
        };
        if (op == "GGML_OP_NONE") {
            // A weight leaf carries the GGUF tensor name as the NODE name, not as an input.
            auto w = m_w->find(name);
            return w == m_w->end() ? nullptr : w->second;
        }
        if (op == "GGML_OP_GET_ROWS")
            return ggml_get_rows(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_MUL")
            return ggml_mul(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_ADD")
            return ggml_add(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_MUL_MAT")
            return ggml_mul_mat(m_ctx, get(in, 0), get(in, 1));
        if (op == "GGML_OP_RMS_NORM")
            return ggml_rms_norm(m_ctx, get(in, 0), attr("eps").as<float>());
        if (op == "GGML_OP_SCALE") {
            // Granite-style scalar multipliers (embedding / attention / residual / logits).
            auto s = attr("scale");
            auto b = attr("bias");
            return ggml_scale_bias(m_ctx,
                                   get(in, 0),
                                   s.empty() ? 1.0f : s.as<float>(),
                                   b.empty() ? 0.0f : b.as<float>());
        }
        if (op == "GGML_OP_RESHAPE") {
            auto ne = to_ne(out_shape, m_n_tokens);
            return ggml_reshape_4d(m_ctx, get(in, 0), ne[0], ne[1], ne[2], ne[3]);
        }
        if (op == "GGML_GLU_OP_SWIGLU" || op == "GGML_GLU_OP_GEGLU") {
            ggml_tensor* x = get(in, 0);
            ggml_tensor* y = get(in, 1);
            auto sw = attr("swapped");
            if (!sw.empty() && sw.as<bool>()) {
                std::swap(x, y);
            }
            return op == "GGML_GLU_OP_GEGLU" ? ggml_geglu_split(m_ctx, x, y)
                                             : ggml_swiglu_split(m_ctx, x, y);
        }
        if (op == "GGML_OP_ROPE") {
            const auto rc = attr("rope_config").as<RopeConfig>();
            // The RoPE variant is in the high 16 bits of op_case (arch_registry.hpp:
            // ROPE_OP_CASE_NORMAL/NEOX/IMROPE), mirroring llama_model_rope_type. Getting this
            // wrong is silent: a NEOX arch run as NORMAL still emits fluent but degraded text.
            int mode = GGML_ROPE_TYPE_NORMAL;
            switch (op_case >> 16) {
            case 1:
                mode = GGML_ROPE_TYPE_NEOX;
                break;
            case 2:
                mode = GGML_ROPE_TYPE_IMROPE;
                break;
            default:
                break;
            }
            if (m_rope_mode >= 0) {
                mode = m_rope_mode;
            }
            return ggml_rope_ext(m_ctx,
                                 get(in, 0),
                                 get(in, 1),
                                 get(in, 2),
                                 rc.n_dims,
                                 mode,
                                 rc.n_ctx_orig,
                                 rc.freq_base,
                                 rc.freq_scale,
                                 rc.ext_factor,
                                 rc.attn_factor,
                                 rc.beta_fast,
                                 rc.beta_slow);
        }
        if (op == "GGML_OP_SET_ROWS") {
            // ggml_set_rows wants rows, so collapse [head_size, n_head_kv, n_kv] to
            // [head_size*n_head_kv, n_kv] for the write and restore the 4-D view after.
            ggml_tensor* d = get(in, 0);
            ggml_tensor* idx = get(in, 1);
            ggml_tensor* c = get(in, 2);
            ggml_tensor* src = ggml_reshape_4d(m_ctx, d, d->ne[0] * d->ne[1], d->ne[2], d->ne[3], 1);
            ggml_tensor* dst = ggml_reshape_4d(m_ctx, c, c->ne[0] * c->ne[1], c->ne[2], c->ne[3], 1);
            ggml_tensor* r = ggml_set_rows(m_ctx, dst, src, idx);
            return ggml_reshape_4d(m_ctx, r, c->ne[0], c->ne[1], c->ne[2], c->ne[3]);
        }
        if (op == "GGML_OP_FLASH_ATTN_EXT") {
            auto q = ggml_cont(m_ctx, ggml_permute(m_ctx, get(in, 0), 0, 2, 1, 3));
            auto k = ggml_cont(m_ctx, ggml_permute(m_ctx, get(in, 1), 0, 2, 1, 3));
            auto v = ggml_cont(m_ctx, ggml_permute(m_ctx, get(in, 2), 0, 2, 1, 3));
            auto sc = attr("scale");
            auto cap = attr("kq_soft_cap");
            return ggml_flash_attn_ext(m_ctx,
                                       q,
                                       k,
                                       v,
                                       get(in, 3),
                                       sc.empty() ? 1.0f : sc.as<float>(),
                                       0.0f,
                                       cap.empty() ? 0.0f : cap.as<float>());
        }
        return nullptr;  // recorded in unsupported_ops()
    }

    ggml_context* m_ctx = nullptr;
    ggml_context* m_ext_ctx = nullptr;
    std::map<std::string, ggml_tensor*>* m_w;
    int m_n_tokens, m_n_kv, m_rope_mode;
    Out* m_out;
};

}  // namespace

const std::set<std::string>& handled_ops() {
    // ADD A CASE TO GgmlEmitter::build() AND A NAME HERE TOGETHER.
    static const std::set<std::string> ops = {
        "GGML_OP_NONE",       "GGML_OP_GET_ROWS", "GGML_OP_MUL",      "GGML_OP_ADD",
        "GGML_OP_MUL_MAT",    "GGML_OP_RMS_NORM", "GGML_OP_SCALE",    "GGML_OP_RESHAPE",
        "GGML_GLU_OP_SWIGLU", "GGML_GLU_OP_GEGLU", "GGML_OP_ROPE",    "GGML_OP_SET_ROWS",
        "GGML_OP_FLASH_ATTN_EXT",
    };
    return ops;
}

struct GgmlModel::Impl {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, ebuf = nullptr;
    ggml_context *wctx = nullptr, *ctx_g = nullptr, *ctx_ext = nullptr;
    gguf_context* gg = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    std::map<std::string, ggml_tensor*> wmap;
    Out out;

    ~Impl() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ebuf) ggml_backend_buffer_free(ebuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (ctx_ext) ggml_free(ctx_ext);
        if (ctx_g) ggml_free(ctx_g);
        if (wctx) ggml_free(wctx);
        if (gg) gguf_free(gg);
        if (backend) ggml_backend_free(backend);
    }
};

GgmlModel::GgmlModel() : m_impl(new Impl()) {}
GgmlModel::~GgmlModel() = default;

std::shared_ptr<GgmlModel> GgmlModel::build(const std::string& gguf_path,
                                            int n_kv,
                                            const std::string& backend_name,
                                            int rope_mode) {
    // The frontend builds a single-token graph; prefill steps one token at a time.
    constexpr int n_tokens = 1;
    std::shared_ptr<GgmlModel> model(new GgmlModel());
    auto& im = *model->m_impl;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    if (!backend_name.empty()) {
        dev = ggml_backend_dev_by_name(backend_name.c_str());
        OPENVINO_ASSERT(dev, "[GGML] no such ggml backend device: ", backend_name);
    } else {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    OPENVINO_ASSERT(dev, "[GGML] no ggml backend device available");
    im.backend = ggml_backend_dev_init(dev, nullptr);
    OPENVINO_ASSERT(im.backend, "[GGML] failed to initialise backend");

    // Weights, straight from the file, uploaded to the backend.
    gguf_init_params gp = {true, &im.wctx};
    im.gg = gguf_init_from_file(gguf_path.c_str(), gp);
    OPENVINO_ASSERT(im.gg, "[GGML] cannot read gguf: ", gguf_path);
    im.wbuf = ggml_backend_alloc_ctx_tensors(im.wctx, im.backend);
    FILE* f = fopen(gguf_path.c_str(), "rb");
    OPENVINO_ASSERT(f, "[GGML] cannot open: ", gguf_path);
    const size_t doff = gguf_get_data_offset(im.gg);
    std::vector<uint8_t> tmp;
    for (ggml_tensor* t = ggml_get_first_tensor(im.wctx); t; t = ggml_get_next_tensor(im.wctx, t)) {
        const int i = gguf_find_tensor(im.gg, ggml_get_name(t));
        if (i < 0) {
            continue;
        }
        tmp.resize(ggml_nbytes(t));
        if (fseek(f, static_cast<long>(doff + gguf_get_tensor_offset(im.gg, i)), SEEK_SET) != 0 ||
            fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) {
            fclose(f);
            OPENVINO_THROW("[GGML] short read for tensor ", ggml_get_name(t));
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
        im.wmap[ggml_get_name(t)] = t;
    }
    fclose(f);

    im.ctx_ext = ggml_init({ggml_tensor_overhead() * 512, nullptr, true});
    im.ctx_g = ggml_init({ggml_tensor_overhead() * MAX_NODES +
                              ggml_graph_overhead_custom(MAX_NODES, false),
                          nullptr, true});

    auto factory = [&](std::unordered_map<std::string, ov::Tensor>& w,
                       std::unordered_map<std::string, GgufTensorType>& q,
                       const std::string& arch) -> std::unique_ptr<GraphEmitter> {
        return std::make_unique<GgmlEmitter>(w, q, arch, im.ctx_g, im.ctx_ext, &im.wmap,
                                             n_tokens, n_kv, rope_mode, &im.out);
    };
    build_ggml_graph_from_gguf(gguf_path, factory);

    if (!im.out.unsupported.empty()) {
        std::string list;
        for (const auto& u : im.out.unsupported) {
            list += (list.empty() ? "" : ", ") + u;
        }
        OPENVINO_THROW("[GGML] emitter cannot translate: ", list,
                       ". Add a case to GgmlEmitter::build() and a name to handled_ops().");
    }
    OPENVINO_ASSERT(im.out.last, "[GGML] builder produced no output tensor");

    im.ebuf = ggml_backend_alloc_ctx_tensors(im.ctx_ext, im.backend);
    im.gf = ggml_new_graph_custom(im.ctx_g, MAX_NODES, false);
    ggml_build_forward_expand(im.gf, im.out.last);

    im.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));
    OPENVINO_ASSERT(ggml_gallocr_alloc_graph(im.alloc, im.gf), "[GGML] graph allocation failed");

    // KV caches live in ctx_ext, so they persist across compute() calls; ggml_set_rows writes
    // through a view into that same buffer. Zero them before first use.
    for (const auto& kv : im.out.externals) {
        if (kv.first.rfind("cache_", 0) != 0) {
            continue;
        }
        std::vector<char> z(ggml_nbytes(kv.second), 0);
        ggml_backend_tensor_set(kv.second, z.data(), 0, z.size());
    }
    return model;
}

bool GgmlModel::compute() {
    return ggml_backend_graph_compute(m_impl->backend, m_impl->gf) == GGML_STATUS_SUCCESS;
}

bool GgmlModel::write_input(const std::string& name, const void* data, size_t bytes) {
    auto it = m_impl->out.externals.find(name);
    if (it == m_impl->out.externals.end()) {
        return false;
    }
    OPENVINO_ASSERT(bytes <= ggml_nbytes(it->second),
                    "[GGML] write_input('", name, "'): ", bytes, " bytes exceeds tensor size ",
                    ggml_nbytes(it->second));
    ggml_backend_tensor_set(it->second, data, 0, bytes);
    return true;
}

size_t GgmlModel::input_size(const std::string& name) const {
    auto it = m_impl->out.externals.find(name);
    return it == m_impl->out.externals.end() ? 0 : static_cast<size_t>(ggml_nelements(it->second));
}

std::vector<std::string> GgmlModel::input_names() const {
    std::vector<std::string> names;
    names.reserve(m_impl->out.externals.size());
    for (const auto& kv : m_impl->out.externals) {
        names.push_back(kv.first);
    }
    return names;
}

size_t GgmlModel::logits_size() const {
    return static_cast<size_t>(m_impl->out.last->ne[0]);
}

void GgmlModel::read_logits(float* out) const {
    ggml_backend_tensor_get(m_impl->out.last, out, 0, logits_size() * sizeof(float));
}

const std::map<std::string, ggml_tensor*>& GgmlModel::externals() const {
    return m_impl->out.externals;
}

ggml_tensor* GgmlModel::logits() const {
    return m_impl->out.last;
}

const std::set<std::string>& GgmlModel::unsupported_ops() const {
    return m_impl->out.unsupported;
}

size_t GgmlModel::node_count() const {
    return static_cast<size_t>(ggml_graph_n_nodes(m_impl->gf));
}

std::string GgmlModel::backend_name() const {
    return ggml_backend_name(m_impl->backend);
}

}  // namespace ggml_emitter
}  // namespace ov
