// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Rebuilds a ggml graph from a dumped llama.cpp cgraph artifact (format "ov-cgraph-v1").
//
// This is the graph source for GgmlModel: it replays a topology llama.cpp described offline.
// The executor and the public API (model.cpp) know nothing of llama.cpp -- so llama.cpp is a
// BUILD-TIME tool and never enters the runtime.
//
// TWO THINGS THE ARTIFACT FORMAT FORCED
//
// 1. Identity is by id, not name. ggml graphs identify nodes by POINTER and llama.cpp reuses
//    names (the mul_mat producing Qcur-0 and the rope consuming it are both "Qcur-0"), so the
//    dumper interns pointers to unique ids and we resolve inputs through those.
//
// 2. Inputs are identified by ROLE, not name. llama.cpp leaves most graph inputs ggml-autonamed
//    ("leaf_5"), so their identity is recovered from which op consumes them -- ROPE[src1] is
//    positions, SET_ROWS[src1] is a cache row index, and so on. The canonical names are then
//    published through externals() so callers bind inputs by role, not by leaf name.
//
// SHAPE-STATIC: the artifact records n_tokens and n_kv because they are baked into every node's
// dimensions. A mismatch is rejected rather than silently producing nonsense.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model_impl.hpp"

namespace ov {
namespace ggml_cgraph_loader {

namespace {

using json = nlohmann::json;

ggml_type type_from_name(const std::string& n) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const auto t = static_cast<ggml_type>(i);
        const char* tn = ggml_type_name(t);
        if (tn && n == tn) {
            return t;
        }
    }
    OPENVINO_THROW("[GGML] unknown tensor type in artifact: ", n);
}

// op_params is the raw ggml blob; each op owns its layout. Decoding lives here so exactly one
// place knows it, mirroring how ggml itself writes the params.
float f_param(const json& n, size_t i, float dflt = 0.0f) {
    if (!n.contains("op_params") || n["op_params"].size() <= i) {
        return dflt;
    }
    const int32_t raw = n["op_params"][i].get<int32_t>();
    float f;
    std::memcpy(&f, &raw, sizeof(f));
    return f;
}

int32_t i_param(const json& n, size_t i, int32_t dflt = 0) {
    if (!n.contains("op_params") || n["op_params"].size() <= i) {
        return dflt;
    }
    return n["op_params"][i].get<int32_t>();
}

std::vector<uint8_t> from_base64(const std::string& in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        return c == '+' ? 62 : c == '/' ? 63 : -1;
    };
    std::vector<uint8_t> out;
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        const int v = val(c);
        if (v < 0) {
            continue;  // '=' padding
        }
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

struct Loader {
    GgmlModel::Impl& im;
    std::map<std::string, ggml_tensor*> by_id;

    // The base model's KV cache of this name if this graph is built over one: the cache itself,
    // or a view of its first ne[1] slots for a graph dumped with a smaller n_kv.
    ggml_tensor* base_cache(const std::string& name, ggml_type type, const std::vector<int64_t>& ne) const {
        if (!im.base || name.rfind("cache_", 0) != 0) {
            return nullptr;
        }
        const auto& ext = im.base->externals();
        auto it = ext.find(name);
        if (it == ext.end() || it->second->type != type) {
            return nullptr;
        }
        ggml_tensor* c = it->second;
        if (c->ne[0] != ne[0] || c->ne[2] != ne[2] || c->ne[3] != ne[3] || c->ne[1] < ne[1]) {
            return nullptr;
        }
        if (c->ne[1] == ne[1]) {
            return c;
        }
        return ggml_view_4d(im.ctx_g, c, ne[0], ne[1], ne[2], ne[3], c->nb[1], c->nb[2], c->nb[3], 0);
    }

    ggml_tensor* get(const std::string& id) const {
        auto it = by_id.find(id);
        if (it == by_id.end() && !im.out.unsupported.empty()) {
            // The id belongs to a node skipped as untranslatable; name the ops instead.
            im.throw_if_unsupported();
        }
        OPENVINO_ASSERT(it != by_id.end(), "[GGML] artifact references unknown tensor id: ", id);
        return it->second;
    }
    ggml_tensor* in(const json& n, size_t i) const {
        OPENVINO_ASSERT(n["inputs"].size() > i, "[GGML] node ", n["id"].get<std::string>(),
                        " (", n["op"].get<std::string>(), ") needs input ", i);
        return get(n["inputs"][i].get<std::string>());
    }

    // Create the graph leaves: weights resolved from the .gguf, plus the inputs and KV caches,
    // which go in ctx_ext so they survive across compute() calls.
    void build_leaves(const json& doc) {
        const auto& weights = doc["weights"];
        for (const auto& l : doc["leaves"]) {
            const std::string id = l["id"];
            if (weights.contains(id)) {
                const std::string gname = weights[id]["gguf_name"];
                auto w = im.wmap.find(gname);
                if (w == im.wmap.end()) {
                    // stable-diffusion.cpp prefixes its parameter names ("model.diffusion_model.") where the
                    // GGUF stores them bare
                    static const char* prefixes[] = {"model.diffusion_model.", "diffusion_model.", "model."};
                    for (const char* pre : prefixes) {
                        if (gname.rfind(pre, 0) == 0) {
                            w = im.wmap.find(gname.substr(strlen(pre)));
                            if (w != im.wmap.end()) {
                                break;
                            }
                        }
                    }
                }
                if (w == im.wmap.end() && gname.rfind("leaf_", 0) == 0) {
                    // An unnamed buffer-backed leaf is a scratch tensor llama.cpp allocated, e.g. the shape
                    // template of a GGML_OP_FILL, whose contents the op never reads: zeros do.
                    const auto lne = l["ne"].get<std::vector<int64_t>>();
                    ggml_tensor* t = ggml_new_tensor_4d(im.ctx_ext, type_from_name(l["type"]), lne[0], lne[1], lne[2], lne[3]);
                    ggml_set_name(t, gname.c_str());
                    im.constants.emplace_back(t, std::vector<uint8_t>(ggml_nbytes(t), 0));
                    by_id[id] = t;
                    continue;
                }
                OPENVINO_ASSERT(w != im.wmap.end(),
                                "[GGML] artifact expects weight '", gname,
                                "' which is not in the model file -- artifact and .gguf disagree");
                ggml_tensor* wt = w->second;
                // llama.cpp may give a weight a different shape than the file stores (a [n_embd * hc] vector
                // used as [n_embd, hc]); the artifact records the shape the graph used.
                const auto lne = l["ne"].get<std::vector<int64_t>>();
                if (lne.size() == 4 && (wt->ne[0] != lne[0] || wt->ne[1] != lne[1] || wt->ne[2] != lne[2] || wt->ne[3] != lne[3]) &&
                    ggml_nelements(wt) == lne[0] * lne[1] * lne[2] * lne[3] && ggml_is_contiguous(wt)) {
                    wt = ggml_reshape_4d(im.ctx_g, wt, lne[0], lne[1], lne[2], lne[3]);
                }
                by_id[id] = wt;
                continue;
            }
            const auto ne = l["ne"].get<std::vector<int64_t>>();
            const std::string name = l["name"];
            if (ggml_tensor* shared = base_cache(name, type_from_name(l["type"]), ne)) {
                by_id[id] = shared;
                im.shared_caches.insert(shared);
                im.out.externals[name] = shared;
                continue;
            }
            ggml_tensor* t = ggml_new_tensor_4d(im.ctx_ext, type_from_name(l["type"]), ne[0], ne[1],
                                                ne[2], ne[3]);
            ggml_set_name(t, name.c_str());
            by_id[id] = t;
            // A constant input carries its value in the artifact (e.g. llama.cpp's KV-cache
            // Hadamard rotation matrices): uploaded once in finalize(), not a caller input.
            if (l.contains("data_b64")) {
                auto data = from_base64(l["data_b64"].get<std::string>());
                OPENVINO_ASSERT(data.size() == ggml_nbytes(t), "[GGML] constant input '", name, "' has ",
                                data.size(), " bytes, expected ", ggml_nbytes(t));
                im.constants.emplace_back(t, std::move(data));
                continue;
            }
            if (l["is_input"].get<bool>()) {
                ggml_set_input(t);
            } else if (l["has_buffer"].get<bool>() && name.rfind("cache_k", 0) != 0 && name.rfind("cache_v", 0) != 0) {
                im.out.state_names.insert(name);
            }
            im.out.externals[name] = t;
        }
    }

    // VIEW / RESHAPE / PERMUTE / TRANSPOSE are metadata-only: nothing is computed, consumers just
    // read through nb[]. So instead of replaying each op's semantics we rebuild the exact
    // (view_src, offset, ne, nb) that was recorded -- one path for four op types, and it cannot
    // disagree with the original because it IS the original metadata.
    ggml_tensor* rebuild_view(const json& n) {
        ggml_tensor* src = get(n["view_src"].get<std::string>());
        const auto ne = n["ne"].get<std::vector<int64_t>>();
        const auto nb = n["nb"].get<std::vector<size_t>>();
        ggml_tensor* t = ggml_view_4d(im.ctx_g, src, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2],
                                      nb[3], n["view_offs"].get<size_t>());
        // ggml_view_4d derives nb[0] from the type; a permuted or transposed view has a different
        // nb[0], so restore all four verbatim.
        for (int i = 0; i < 4; i++) {
            t->nb[i] = nb[i];
        }
        return t;
    }

    // The CPU element-wise kernels take F32 / F16 second operands; a BF16 weight (a norm scale) is widened first.
    static ggml_tensor* operand(ggml_context* c, ggml_tensor* b) {
        return b->type == GGML_TYPE_BF16 ? ggml_cast(c, b, GGML_TYPE_F32) : b;
    }

    ggml_tensor* build_node(const json& n) {
        const std::string op = n["op"];
        ggml_context* c = im.ctx_g;

        if (op == "GGML_OP_GET_ROWS")  return ggml_get_rows(c, in(n, 0), in(n, 1));
        if (op == "GGML_OP_MUL") {
            ggml_tensor* a = in(n, 0);
            ggml_tensor* b = operand(c, in(n, 1));
            OPENVINO_ASSERT(ggml_can_repeat(b, a), "[GGML] MUL ", n["name"].get<std::string>(), ": operand ne=[", b->ne[0], ",",
                            b->ne[1], ",", b->ne[2], ",", b->ne[3], "] cannot broadcast into [", a->ne[0], ",", a->ne[1], ",",
                            a->ne[2], ",", a->ne[3], "] (artifact recorded ", n["input_ne"].dump(), ")");
            return ggml_mul(c, a, b);
        }
        if (op == "GGML_OP_ADD")       return ggml_add(c, in(n, 0), operand(c, in(n, 1)));
        if (op == "GGML_OP_MUL_MAT") {
            // op_params: [0] precision, [1] hint. The hint matters: llama.cpp marks its quantized-KV
            // Hadamard rotations SRC0_IS_HADAMARD, which backends run as a fast Walsh-Hadamard
            // transform. That is the same math but rounds differently from a dense matmul, enough to
            // move values across quantization steps once they are stored in a q8_0/q4_0 cache.
            ggml_tensor* r = ggml_mul_mat(c, in(n, 0), in(n, 1));
            if (const int32_t prec = i_param(n, 0)) {
                ggml_mul_mat_set_prec(r, static_cast<ggml_prec>(prec));
            }
            if (const int32_t hint = i_param(n, 1)) {
                ggml_mul_mat_set_hint(r, static_cast<ggml_op_hint>(hint));
            }
            return r;
        }
        if (op == "GGML_OP_RMS_NORM")  return ggml_rms_norm(c, in(n, 0), f_param(n, 0, 1e-5f));
        if (op == "GGML_OP_SCALE") {
            ggml_tensor* x = in(n, 0);
            if (x->type != GGML_TYPE_F32) {  // the CPU kernel scales F32 only; go through F32 and back
                ggml_tensor* y = ggml_scale_bias(c, ggml_cast(c, x, GGML_TYPE_F32), f_param(n, 0, 0.0f), f_param(n, 1));
                return ggml_cast(c, y, x->type);
            }
            return ggml_scale_bias(c, x, f_param(n, 0, 0.0f), f_param(n, 1));
        }
        if (op == "GGML_OP_CONT") {
            // ggml_cont_2d/3d/4d make a tensor contiguous AND reshape it -- llama.cpp merges
            // attention heads that way (12 heads x 32 -> 384). Plain ggml_cont would keep the
            // source shape, so use the shape the artifact recorded.
            const auto ne = n["ne"].get<std::vector<int64_t>>();
            return ggml_cont_4d(c, in(n, 0), ne[0], ne[1], ne[2], ne[3]);
        }
        if (op == "GGML_OP_SILU")      return ggml_silu(c, in(n, 0));
        // Vision towers prepend a class embedding to the patch embeddings. op_params[0] = dim.
        if (op == "GGML_OP_CONCAT")    return ggml_concat(c, in(n, 0), in(n, 1), i_param(n, 0));
        if (op == "GGML_OP_REPEAT") {
            // ggml_repeat keeps only its source; the target shape is the recorded output shape.
            const auto ne = n["ne"].get<std::vector<int64_t>>();
            return ggml_repeat_4d(c, in(n, 0), ne[0], ne[1], ne[2], ne[3]);
        }
        // Vision towers: the patch embedding is a convolution, which ggml expresses as
        // im2col + mul_mat. src0 is the kernel, src1 the image.
        if (op == "GGML_OP_IM2COL") {
            return ggml_im2col(c, in(n, 0), in(n, 1), i_param(n, 0, 1), i_param(n, 1, 1),
                               i_param(n, 2), i_param(n, 3), i_param(n, 4, 1), i_param(n, 5, 1),
                               i_param(n, 6) != 0, type_from_name(n["type"]));
        }
        if (op == "GGML_OP_CPY") {
            // ggml_cpy(a, b): b is a real destination (already built -- often a cache view).
            // ggml_cast(a, type): SELF-REFERENCING by design -- result->src[1] = result, per a
            // comment in ggml.c ("needed by some backends for consistency with ggml_cpy_impl").
            // The artifact preserves that: this node's own id appears in its own inputs. That
            // can only be ggml_cast (a real dst tensor cannot be its own producer), so build the
            // cast and let the artifact's own ne/type describe the destination.
            const std::string& dst_id = n["inputs"][1].get<std::string>();
            if (dst_id == n["id"].get<std::string>()) {
                return ggml_cast(c, in(n, 0), type_from_name(n["type"]));
            }
            return ggml_cpy(c, in(n, 0), in(n, 1));
        }
        // LayerNorm. Encoder (BERT-family) models use this where decoders use RMS_NORM.
        if (op == "GGML_OP_NORM")      return ggml_norm(c, in(n, 0), f_param(n, 0, 1e-5f));
        if (op == "GGML_OP_SOFT_MAX") {
            // Explicit (non-flash) attention passes a mask and a scale; a bare softmax has
            // neither. Encoders take this path because their attention is bidirectional.
            if (n["inputs"].size() > 1) {
                if (f_param(n, 1) > 0.0f) im.alibi = true;
                return ggml_soft_max_ext(c, in(n, 0), in(n, 1), f_param(n, 0, 1.0f),
                                         f_param(n, 1));
            }
            return ggml_soft_max(c, in(n, 0));
        }
        // Unary activations. op_params[0] holds the ggml_unary_op enum, which the dumper has
        // already resolved into the op name, so nothing more needs decoding here.
        if (op == "GGML_UNARY_OP_GELU")       return ggml_gelu(c, in(n, 0));
        if (op == "GGML_UNARY_OP_GELU_QUICK") return ggml_gelu_quick(c, in(n, 0));
        if (op == "GGML_UNARY_OP_RELU")       return ggml_relu(c, in(n, 0));
        if (op == "GGML_UNARY_OP_SILU")       return ggml_silu(c, in(n, 0));
        if (op == "GGML_UNARY_OP_TANH")       return ggml_tanh(c, in(n, 0));

        if (op == "GGML_OP_SET_ROWS") {
            // src order is (values, indices, dst) -- see ggml.c, where the comment calls it
            // "weird due to legacy reasons". dst is also the view_src of the result.
            return ggml_set_rows(c, in(n, 2), in(n, 0), in(n, 1));
        }
        if (op == "GGML_GLU_OP_SWIGLU" || op == "GGML_GLU_OP_GEGLU") {
            ggml_tensor* a = in(n, 0);
            // A split GLU has two sources; a fused one has a single doubled-width source.
            ggml_tensor* b = n["inputs"].size() > 1 ? in(n, 1) : nullptr;
            const bool swapped = i_param(n, 1) != 0;
            if (b && swapped) {
                std::swap(a, b);
            }
            if (op == "GGML_GLU_OP_GEGLU") {
                return b ? ggml_geglu_split(c, a, b) : ggml_geglu(c, a);
            }
            return b ? ggml_swiglu_split(c, a, b) : ggml_swiglu(c, a);
        }
        if (op == "GGML_OP_ROPE") {
            // params: [1]=n_dims [2]=mode [4]=n_ctx_orig, floats at [5..10], M-RoPE sections [11..14].
            const int mode = i_param(n, 2);
            ggml_tensor* freq = n["inputs"].size() > 2 ? in(n, 2) : nullptr;
            if (mode & GGML_ROPE_TYPE_MROPE) {  // MROPE, IMROPE and VISION all carry this bit
                int sections[GGML_MROPE_SECTIONS];
                for (int k = 0; k < GGML_MROPE_SECTIONS; k++) {
                    sections[k] = i_param(n, 11 + k);
                }
                ggml_tensor* rm = ggml_rope_multi(c, in(n, 0), in(n, 1), freq, i_param(n, 1), sections, mode,
                                                  i_param(n, 4), f_param(n, 5, 10000.0f), f_param(n, 6, 1.0f),
                                                  f_param(n, 7), f_param(n, 8, 1.0f), f_param(n, 9, 32.0f),
                                                  f_param(n, 10, 1.0f));
                if (const int offs = i_param(n, 15)) {
                    ggml_rope_set_offset(rm, offs);
                }
                return rm;
            }
            ggml_tensor* r = ggml_rope_ext(c, in(n, 0), in(n, 1), freq,
                                           i_param(n, 1), mode, i_param(n, 4),
                                           f_param(n, 5, 10000.0f), f_param(n, 6, 1.0f), f_param(n, 7),
                                           f_param(n, 8, 1.0f), f_param(n, 9, 32.0f), f_param(n, 10, 1.0f));
            // params[15] = n_offs: the rotated dims start at this offset (MLA models rotate only the
            // trailing rope part of a [nope | rope] head).
            if (const int offs = i_param(n, 15)) {
                ggml_rope_set_offset(r, offs);
            }
            return r;
        }
        if (op == "GGML_OP_FLASH_ATTN_EXT") {
            // floats [0..2] = scale, max_bias, logit_softcap.
            if (f_param(n, 1) > 0.0f) im.alibi = true;
            ggml_tensor* r = ggml_flash_attn_ext(c, in(n, 0), in(n, 1), in(n, 2),
                                                 n["inputs"].size() > 3 ? in(n, 3) : nullptr,
                                                 f_param(n, 0, 1.0f), f_param(n, 1),
                                                 f_param(n, 2));
            // The dumped graph records FA's output type; llama.cpp sets it explicitly and the
            // default does not always match.
            ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
            if (n["inputs"].size() > 4) {
                ggml_flash_attn_ext_add_sinks(r, in(n, 4));  // per-head attention sinks (gpt-oss)
            }
            return r;
        }
        return build_more(n, op, c);
    }

    // Ops beyond the core decoder set: elementwise math, MoE routing, normalization variants,
    // padding/shape utilities and the recurrent (SSM / RWKV / linear attention) kernels. Parameters
    // are read from op_params exactly where ggml.c stores them.
    ggml_tensor* build_more(const json& n, const std::string& op, ggml_context* c) {
        const size_t n_in = n["inputs"].size();
        auto ne = [&](int d) {
            return n["ne"][d].get<int64_t>();
        };
        // elementwise
        if (op == "GGML_OP_SUB")       return ggml_sub(c, in(n, 0), operand(c, in(n, 1)));
        if (op == "GGML_OP_DIV")       return ggml_div(c, in(n, 0), operand(c, in(n, 1)));
        if (op == "GGML_OP_SQR")       return ggml_sqr(c, in(n, 0));
        if (op == "GGML_OP_SQRT")      return ggml_sqrt(c, in(n, 0));
        if (op == "GGML_OP_LOG")       return ggml_log(c, in(n, 0));
        if (op == "GGML_OP_SIN")       return ggml_sin(c, in(n, 0));
        if (op == "GGML_OP_COS")       return ggml_cos(c, in(n, 0));
        if (op == "GGML_OP_DUP")       return ggml_dup(c, in(n, 0));
        if (op == "GGML_OP_CLAMP")     return ggml_clamp(c, in(n, 0), f_param(n, 0), f_param(n, 1));
        if (op == "GGML_OP_LEAKY_RELU") return ggml_leaky_relu(c, in(n, 0), f_param(n, 0), false);
        if (op == "GGML_OP_ADD_ID")    return ggml_add_id(c, in(n, 0), in(n, 1), in(n, 2));
        // reductions / indexing
        if (op == "GGML_OP_SUM")       return ggml_sum(c, in(n, 0));
        if (op == "GGML_OP_SUM_ROWS")  return ggml_sum_rows(c, in(n, 0));
        if (op == "GGML_OP_MEAN")      return ggml_mean(c, in(n, 0));
        if (op == "GGML_OP_ARGMAX")    return ggml_argmax(c, in(n, 0));
        if (op == "GGML_OP_CUMSUM")    return ggml_cumsum(c, in(n, 0));
        if (op == "GGML_OP_ARGSORT")   return ggml_argsort(c, in(n, 0), static_cast<ggml_sort_order>(i_param(n, 0)));
        if (op == "GGML_OP_TOP_K")     return ggml_top_k(c, in(n, 0), static_cast<int>(ne(0)));
        if (op == "GGML_OP_MUL_MAT_ID") return ggml_mul_mat_id(c, in(n, 0), in(n, 1), in(n, 2));
        // normalization
        if (op == "GGML_OP_L2_NORM")   return ggml_l2_norm(c, in(n, 0), f_param(n, 0));
        if (op == "GGML_OP_GROUP_NORM") return ggml_group_norm(c, in(n, 0), i_param(n, 0), f_param(n, 1));
        // shape / layout
        if (op == "GGML_OP_PAD") {
            return ggml_pad_ext(c, in(n, 0), i_param(n, 0), i_param(n, 1), i_param(n, 2), i_param(n, 3),
                                i_param(n, 4), i_param(n, 5), i_param(n, 6), i_param(n, 7));
        }
        if (op == "GGML_OP_ROLL") {
            return ggml_roll(c, in(n, 0), i_param(n, 0), i_param(n, 1), i_param(n, 2), i_param(n, 3));
        }
        if (op == "GGML_OP_DIAG_MASK_INF") return ggml_diag_mask_inf(c, in(n, 0), i_param(n, 0));
        if (op == "GGML_OP_TRI")       return ggml_tri(c, in(n, 0), static_cast<ggml_tri_type>(i_param(n, 0)));
        if (op == "GGML_OP_FILL")      return ggml_fill(c, in(n, 0), f_param(n, 0));
        if (op == "GGML_OP_SOLVE_TRI") return ggml_solve_tri(c, in(n, 0), in(n, 1), true, true, false);
        if (op == "GGML_OP_SET") {
            return ggml_set(c, in(n, 0), in(n, 1), i_param(n, 0), i_param(n, 1), i_param(n, 2), i_param(n, 3));
        }
        if (op == "GGML_OP_ACC") {
            return ggml_acc(c, in(n, 0), in(n, 1), i_param(n, 0), i_param(n, 1), i_param(n, 2), i_param(n, 3));
        }
        if (op == "GGML_OP_TIMESTEP_EMBEDDING") {
            return ggml_timestep_embedding(c, in(n, 0), i_param(n, 0), i_param(n, 1));
        }
        if (op == "GGML_OP_ROPE_APPLY") {
            // stable-diffusion.cpp's rotary apply: x [D, H, T, B] with per-token 2x2 rotations pe [2, 2, D/2, T]
            // -> [D, T, H*B]. pe[0,0] = cos, pe[0,1] = sin; interleaved pairs also carry pe[1,0] = -sin, pe[1,1] = cos.
            ggml_tensor* x = in(n, 0);
            ggml_tensor* pe = in(n, 1);
            const bool interleaved = i_param(n, 0) != 0;
            const int64_t D = x->ne[0], H = x->ne[1], T = x->ne[2], B = x->ne[3], half = D / 2;
            ggml_tensor* xp = ggml_permute(c, x, 0, 2, 1, 3);  // [D, T, H, B]
            auto pe_part = [&](size_t offset) {  // [half, T, 1, 1]: element (i, j, t) of pe at i = offset
                ggml_tensor* v = ggml_view_4d(c, pe, half, T, 1, 1, pe->nb[3], pe->nb[3] * T, pe->nb[3] * T, offset);
                v->nb[0] = pe->nb[2];
                return ggml_cont(c, v);
            };
            ggml_tensor* cos_t = pe_part(0);
            ggml_tensor* sin_t = pe_part(pe->nb[1]);
            auto part = [&](size_t offset) {  // elements [offset, offset + half) along dim 0, as [half, T, H, B]
                return ggml_cont(c, ggml_view_4d(c, xp, half, T, H, B, xp->nb[1], xp->nb[2], xp->nb[3], offset * xp->nb[0]));
            };
            ggml_tensor *a, *b, *o0, *o1;
            if (interleaved) {
                auto strided = [&](size_t offset) {
                    ggml_tensor* v = ggml_view_4d(c, xp, half, T, H, B, xp->nb[1], xp->nb[2], xp->nb[3], offset * xp->nb[0]);
                    v->nb[0] = 2 * xp->nb[0];
                    return ggml_cont(c, v);
                };
                a = strided(0);
                b = strided(1);
                ggml_tensor* neg_sin = pe_part(pe->nb[0]);
                ggml_tensor* cos1 = pe_part(pe->nb[0] + pe->nb[1]);
                o0 = ggml_add(c, ggml_mul(c, a, cos_t), ggml_mul(c, b, neg_sin));
                o1 = ggml_add(c, ggml_mul(c, a, sin_t), ggml_mul(c, b, cos1));
                ggml_tensor* both = ggml_concat(c, ggml_reshape_4d(c, o0, 1, half, T, H * B), ggml_reshape_4d(c, o1, 1, half, T, H * B), 0);
                return ggml_reshape_3d(c, both, D, T, H * B);
            }
            a = part(0);
            b = part(half);
            o0 = ggml_sub(c, ggml_mul(c, a, cos_t), ggml_mul(c, b, sin_t));
            o1 = ggml_add(c, ggml_mul(c, a, sin_t), ggml_mul(c, b, cos_t));
            return ggml_reshape_3d(c, ggml_concat(c, o0, o1, 0), D, T, H * B);
        }
        if (op == "GGML_OP_LIGHTNING_INDEXER") {
            // DSA / QSA indexer score, in plain ops as llama.cpp builds it without the fused op:
            // sum_h relu(k . q_h) * w_h + mask, [n_kv, n_batch].
            ggml_tensor* qp = ggml_permute(c, in(n, 0), 0, 2, 1, 3);
            ggml_tensor* kp = ggml_permute(c, in(n, 1), 0, 2, 1, 3);
            ggml_tensor* kq = ggml_mul_mat(c, kp, qp);
            kq = ggml_cont(c, ggml_permute(c, kq, 2, 1, 0, 3));
            ggml_tensor* score = ggml_mul(c, ggml_relu(c, kq), in(n, 2));
            score = ggml_cont(c, ggml_permute(c, ggml_sum_rows(c, score), 2, 1, 0, 3));
            ggml_tensor* mask = in(n, 3);
            return ggml_add(c, score, mask->type == GGML_TYPE_F32 ? mask : ggml_cast(c, mask, GGML_TYPE_F32));
        }
        if (op == "GGML_OP_DSV4_HC_PRE") {
            // hyper-connection pre-mix: x [E, hc, T] weighted over the hc streams -> [E, T]; ggml has no such op
            // in the pinned release, so it is built from plain ops. op_params: f32 scale, i32 gated.
            ggml_tensor* x = in(n, 0);
            ggml_tensor* w = in(n, 1);
            const int64_t E = x->ne[0], hc = x->ne[1], T = x->ne[2];
            ggml_tensor* mixed;
            if (i_param(n, 1)) {  // per-element gate: scale * sum_h x * sigmoid(gate)
                mixed = ggml_mul(c, x, ggml_sigmoid(c, w));
            } else {              // weights [hc, T] broadcast over E
                mixed = ggml_mul(c, x, ggml_reshape_3d(c, w, 1, hc, T));
            }
            mixed = ggml_cont(c, ggml_permute(c, mixed, 1, 0, 2, 3));  // [hc, E, T]
            ggml_tensor* sum = ggml_reshape_2d(c, ggml_sum_rows(c, mixed), E, T);
            const float scale = f_param(n, 0, 1.0f);
            return scale == 1.0f ? sum : ggml_scale(c, sum, scale);
        }
        // recurrent kernels
        if (op == "GGML_OP_SSM_CONV")  return ggml_ssm_conv(c, in(n, 0), in(n, 1));
        if (op == "GGML_OP_SSM_SCAN") {
            // op_params[0] = K, the number of states kept per sequence
            return ggml_ssm_scan(c, in(n, 0), in(n, 1), in(n, 2), in(n, 3), in(n, 4), in(n, 5), in(n, 6),
                                 std::max(1, i_param(n, 0)));
        }
        if (op == "GGML_OP_RWKV_WKV6") {
            return ggml_rwkv_wkv6(c, in(n, 0), in(n, 1), in(n, 2), in(n, 3), in(n, 4), in(n, 5));
        }
        if (op == "GGML_OP_RWKV_WKV7") {
            return ggml_rwkv_wkv7(c, in(n, 0), in(n, 1), in(n, 2), in(n, 3), in(n, 4), in(n, 5), in(n, 6));
        }
        if (op == "GGML_OP_GATED_DELTA_NET") {
            // q, k, v, g, beta, state; op_params[0] = K (state snapshots kept)
            return ggml_gated_delta_net(c, in(n, 0), in(n, 1), in(n, 2), in(n, 3), in(n, 4), in(n, 5),
                                        std::max(1, i_param(n, 0)));
        }
        if (op == "GGML_OP_GATED_LINEAR_ATTN") {
            return ggml_gated_linear_attn(c, in(n, 0), in(n, 1), in(n, 2), in(n, 3), in(n, 4), f_param(n, 0));
        }
        // activations not covered above: any unary op by name, xIELU with its fitted constants,
        // and the remaining gated linear units
        if (op.rfind("GGML_UNARY_OP_", 0) == 0) {
            const std::string name = op.substr(std::string("GGML_UNARY_OP_").size());
            if (name == "XIELU") {
                // op_params hold beta + softplus(alpha_n), softplus(alpha_p), beta, eps; invert softplus.
                const float beta = f_param(n, 3), eps = f_param(n, 4);
                // softplus(x) = log(1 + e^x); its inverse y + log(1 - e^-y) stays finite for the large
                // values (hundreds) these models store, where log(expm1(y)) overflows.
                auto inv_softplus = [](float y) {
                    return static_cast<float>(y + std::log(-std::expm1(-static_cast<double>(y))));
                };
                const float an = inv_softplus(f_param(n, 1) - beta), ap = inv_softplus(f_param(n, 2));
                return ggml_xielu(c, in(n, 0), an, ap, beta, eps);
            }
            for (int u = 0; u < GGML_UNARY_OP_COUNT; u++) {
                if (name == ggml_unary_op_name(static_cast<ggml_unary_op>(u))) {
                    return ggml_unary(c, in(n, 0), static_cast<ggml_unary_op>(u));
                }
            }
            return nullptr;
        }
        if (op.rfind("GGML_GLU_OP_", 0) == 0) {
            const std::string name = op.substr(std::string("GGML_GLU_OP_").size());
            const bool swapped = i_param(n, 1) != 0;
            if (name == "SWIGLU_CLAMP") {
                OPENVINO_ASSERT(n_in > 1, "[GGML] fused SWIGLU_CLAMP is not supported");
                return ggml_swiglu_clamp(c, in(n, 0), in(n, 1), f_param(n, 3));
            }
            if (name == "SWIGLU_OAI") {
                ggml_tensor* a = in(n, 0);
                ggml_tensor* b = n_in > 1 ? in(n, 1) : nullptr;
                OPENVINO_ASSERT(b, "[GGML] fused SWIGLU_OAI is not supported");
                return ggml_swiglu_oai(c, swapped ? b : a, swapped ? a : b, f_param(n, 2), f_param(n, 3));
            }
            for (int g = 0; g < GGML_GLU_OP_COUNT; g++) {
                if (name == ggml_glu_op_name(static_cast<ggml_glu_op>(g))) {
                    const auto gop = static_cast<ggml_glu_op>(g);
                    if (n_in > 1) {
                        return swapped ? ggml_glu_split(c, in(n, 1), in(n, 0), gop) : ggml_glu_split(c, in(n, 0), in(n, 1), gop);
                    }
                    return ggml_glu(c, in(n, 0), gop, swapped);
                }
            }
            return nullptr;
        }
        return nullptr;
    }

    void build_nodes(const json& doc) {
        for (const auto& n : doc["nodes"]) {
            const std::string id = n["id"];
            const std::string op = n["op"];
            ggml_tensor* t = nullptr;

            // Only the metadata-only ops take the view shortcut. Real ops can carry a view_src
            // too -- SET_ROWS and CPY both return a view of their destination -- and rebuilding
            // those as a bare view silently drops the write (whisper stores its cross-attention
            // K/V with CPY).
            const bool metadata_only = op == "GGML_OP_VIEW" || op == "GGML_OP_RESHAPE" ||
                                       op == "GGML_OP_PERMUTE" || op == "GGML_OP_TRANSPOSE";
            if (metadata_only && n.contains("view_src")) {
                t = rebuild_view(n);
            } else {
                t = build_node(n);
            }
            if (!t) {
                im.out.unsupported.insert(op);
                continue;
            }
            // The artifact records what llama.cpp actually produced, so any divergence here is
            // a reconstruction bug -- catch it at the node that caused it rather than letting a
            // downstream op assert on operands whose provenance is no longer obvious.
            const auto want = n["ne"].get<std::vector<int64_t>>();
            for (int d = 0; d < 4; d++) {
                OPENVINO_ASSERT(t->ne[d] == want[d],
                                "[GGML] rebuilt node '", id, "' (", op, ") has ne[", d, "]=",
                                t->ne[d], " but the artifact recorded ", want[d],
                                " -- reconstruction does not match the dumped graph");
            }
            const std::string name = n["name"];
            ggml_set_name(t, name.empty() ? id.c_str() : name.c_str());
            if (n.value("is_output", false) && !name.empty()) {
                ggml_set_output(t);  // keeps the allocator from reusing its buffer
                im.out.named_outputs[name] = t;
            }
            by_id[id] = t;
            im.out.map[id] = t;
            // GGML_REF checks every node after the graph ran, so none may share a buffer.
            static const bool keep_all = std::getenv("GGML_REF") != nullptr;
            if (keep_all) ggml_set_output(t);
            im.out.order.push_back(t);
            im.out.last = t;
        }
    }

    // llama.cpp leaves most inputs ggml-autonamed, so recover their identity from the op that
    // consumes them and publish canonical names. Verified against the artifact: ROPE[src1] x32 =
    // positions, SET_ROWS[src1] = cache row index, FLASH_ATTN_EXT[src3] = mask, GET_ROWS[src1] =
    // tokens / out_ids.
    void assign_roles(const json& doc) {
        std::map<std::string, std::string> role;  // leaf id -> canonical name
        int n_setrows = 0, n_getrows = 0, n_masks = 0;
        // Newer llama.cpp names the token input itself; then it, not the first GET_ROWS, is the token input.
        std::string named_tokens_id;
        std::map<std::string, std::string> leaf_name;
        for (const auto& l : doc["leaves"]) {
            leaf_name[l["id"].get<std::string>()] = l["name"].get<std::string>();
            if (l["name"] == "inp_tokens" && l["is_input"].get<bool>()) {
                named_tokens_id = l["id"].get<std::string>();
            }
        }
        // Qwen3.8-Flash-Next's per-layer embedding rows come from an n-gram hash the host computes (an
        // unnamed index input, not a copy of the token ids).
        std::string arch;
        {
            const char* key = "general.architecture";
            const int64_t k = im.gg ? gguf_find_key(im.gg, key) : -1;
            if (k >= 0 && gguf_get_kv_type(im.gg, k) == GGUF_TYPE_STRING) {
                arch = gguf_get_val_str(im.gg, k);
            }
        }
        const bool ngram_ple = im.gg && !arch.empty() && gguf_find_key(im.gg, (arch + ".ple.ngram_size").c_str()) >= 0;
        for (const auto& n : doc["nodes"]) {
            const std::string op = n["op"];
            const auto& ins = n["inputs"];
            auto tag = [&](size_t i, const std::string& canonical) {
                if (ins.size() <= i) {
                    return;
                }
                const std::string id = ins[i];
                if (by_id.count(id) && !role.count(id)) {
                    role[id] = canonical;
                }
            };
            if (op == "GGML_OP_ROPE") {
                tag(1, "inp_pos");
            } else if (op == "GGML_OP_SET_ROWS") {
                // One index leaf per cache (K and V, and again for a sliding-window cache): number
                // them all so each gets its own name; the caller writes every "inp_kv_idx*".
                const std::string id = ins[1];
                if (by_id.count(id) && !role.count(id)) {
                    role[id] = n_setrows == 0 ? "inp_kv_idx" : "inp_kv_idx_" + std::to_string(n_setrows);
                    n_setrows++;
                }
            } else if (op == "GGML_OP_FLASH_ATTN_EXT") {
                // Likewise one mask per cache (full and sliding-window attention).
                if (ins.size() > 3) {
                    const std::string id = ins[3];
                    if (by_id.count(id) && !role.count(id)) {
                        role[id] = n_masks == 0 ? "self_kq_mask" : "self_kq_mask_" + std::to_string(n_masks);
                        n_masks++;
                    }
                }
            } else if (op == "GGML_OP_GET_ROWS") {
                // The first GET_ROWS over a weight (the token embedding) takes the tokens; one
                // over activations selects which rows to emit logits for. Other lookups into
                // weights (e.g. whisper's learned positional embedding) get no canonical name and
                // stay reachable under their own leaf name.
                const std::string id = ins[1];
                const bool from_weight = doc["weights"].contains(ins[0].get<std::string>());
                if (by_id.count(id) && !role.count(id)) {
                    if (!from_weight) {
                        role[id] = "inp_out_ids";
                    } else if (!named_tokens_id.empty() && id == named_tokens_id) {
                        role[id] = "inp_tokens";
                    } else if (ngram_ple && leaf_name[id].rfind("leaf_", 0) == 0) {
                        // left under its own leaf name; the runner computes it
                    } else if (named_tokens_id.empty() && n_getrows++ == 0) {
                        role[id] = "inp_tokens";
                    } else if (!named_tokens_id.empty() && leaf_name[id].rfind("inp_", 0) == 0) {
                        // some other named input (e.g. llama.cpp's mixed token/embedding path)
                    } else {
                        n_getrows += !named_tokens_id.empty();
                        // a further embedding table with its own copy of the token ids (Gemma-3n's
                        // per-layer embeddings): inp_tokens_1, inp_tokens_2, ...
                        role[id] = "inp_tokens_" + std::to_string(n_getrows - 1);
                    }
                }
            }
        }
        for (const auto& l : doc["leaves"]) {
            const std::string id = l["id"];
            if (!by_id.count(id)) {
                continue;
            }
            const std::string name = l["name"];
            if (l["is_input"].get<bool>()) {
                auto r = role.find(id);
                if (r != role.end()) {
                    im.out.externals[r->second] = by_id[id];  // canonical alias
                }
            } else if (l["has_buffer"].get<bool>() && name.rfind("cache_", 0) == 0) {
                im.out.externals[name] = by_id[id];  // finalize() zeroes these
            }
        }
    }
};

}  // namespace

namespace {

json read_artifact(const std::string& cgraph_json) {
    std::ifstream f(cgraph_json);
    OPENVINO_ASSERT(f.good(), "[GGML] cannot open cgraph artifact: ", cgraph_json);
    json doc;
    f >> doc;
    OPENVINO_ASSERT(doc.value("format", "") == "ov-cgraph-v1",
                    "[GGML] unrecognised cgraph artifact format: ", doc.value("format", "<none>"));
    return doc;
}

void build(GgmlModel::Impl& im, const json& doc) {
    im.init_contexts();
    im.n_kv = doc.value("n_kv", 0);

    Loader loader{im, {}};
    loader.build_leaves(doc);
    loader.build_nodes(doc);
    im.throw_if_unsupported();
    loader.assign_roles(doc);
    im.finalize();
}

}  // namespace

std::shared_ptr<GgmlModel> GgmlModel::from_cgraph(const std::string& cgraph_json,
                                                  const std::string& gguf_path,
                                                  const std::string& backend_name) {
    const json doc = read_artifact(cgraph_json);
    std::shared_ptr<GgmlModel> model(new GgmlModel());
    auto& im = *model->m_impl;
    im.init_backend(backend_name);
    im.load_weights(gguf_path);
    build(im, doc);
    return model;
}

std::shared_ptr<GgmlModel> GgmlModel::from_cgraph(const std::string& cgraph_json,
                                                  const std::shared_ptr<GgmlModel>& base) {
    OPENVINO_ASSERT(base, "[GGML] from_cgraph: base model is null");
    const json doc = read_artifact(cgraph_json);
    std::shared_ptr<GgmlModel> model(new GgmlModel());
    model->m_impl->share_from(base, *base->m_impl);
    build(*model->m_impl, doc);
    return model;
}

size_t GgmlModel::context_size() const {
    return m_impl->n_kv;
}

}  // namespace ggml_cgraph_loader
}  // namespace ov
