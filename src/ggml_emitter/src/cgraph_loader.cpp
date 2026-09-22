// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Rebuilds a ggml graph from a dumped llama.cpp cgraph artifact (format "ov-cgraph-v1").
//
// This is the second graph source for GgmlModel. The first (emitter.cpp) has OpenVINO's own
// decoder builder describe the topology; this one replays a topology llama.cpp described
// offline. Same executor, same public API -- so llama.cpp is a BUILD-TIME tool and never enters
// the runtime.
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
//    published through externals() so callers bind inputs identically for both graph sources.
//
// SHAPE-STATIC: the artifact records n_tokens and n_kv because they are baked into every node's
// dimensions. A mismatch is rejected rather than silently producing nonsense.

#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model_impl.hpp"

namespace ov {
namespace ggml_emitter {

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

struct Loader {
    GgmlModel::Impl& im;
    std::map<std::string, ggml_tensor*> by_id;

    ggml_tensor* get(const std::string& id) const {
        auto it = by_id.find(id);
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
                OPENVINO_ASSERT(w != im.wmap.end(),
                                "[GGML] artifact expects weight '", gname,
                                "' which is not in the model file -- artifact and .gguf disagree");
                by_id[id] = w->second;
                continue;
            }
            const auto ne = l["ne"].get<std::vector<int64_t>>();
            ggml_tensor* t = ggml_new_tensor_4d(im.ctx_ext, type_from_name(l["type"]), ne[0], ne[1],
                                                ne[2], ne[3]);
            const std::string name = l["name"];
            ggml_set_name(t, name.c_str());
            if (l["is_input"].get<bool>()) {
                ggml_set_input(t);
            }
            by_id[id] = t;
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

    ggml_tensor* build_node(const json& n) {
        const std::string op = n["op"];
        ggml_context* c = im.ctx_g;

        if (op == "GGML_OP_GET_ROWS")  return ggml_get_rows(c, in(n, 0), in(n, 1));
        if (op == "GGML_OP_MUL")       return ggml_mul(c, in(n, 0), in(n, 1));
        if (op == "GGML_OP_ADD")       return ggml_add(c, in(n, 0), in(n, 1));
        if (op == "GGML_OP_MUL_MAT")   return ggml_mul_mat(c, in(n, 0), in(n, 1));
        if (op == "GGML_OP_RMS_NORM")  return ggml_rms_norm(c, in(n, 0), f_param(n, 0, 1e-5f));
        if (op == "GGML_OP_SCALE")     return ggml_scale_bias(c, in(n, 0), f_param(n, 0, 1.0f),
                                                              f_param(n, 1));
        if (op == "GGML_OP_CONT")      return ggml_cont(c, in(n, 0));
        if (op == "GGML_OP_SOFT_MAX")  return ggml_soft_max(c, in(n, 0));
        if (op == "GGML_OP_SILU")      return ggml_silu(c, in(n, 0));

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
            // params: [1]=n_dims [2]=mode [4]=n_ctx_orig, floats at [5..10].
            return ggml_rope_ext(c, in(n, 0), in(n, 1),
                                 n["inputs"].size() > 2 ? in(n, 2) : nullptr,
                                 i_param(n, 1), i_param(n, 2), i_param(n, 4),
                                 f_param(n, 5, 10000.0f), f_param(n, 6, 1.0f), f_param(n, 7),
                                 f_param(n, 8, 1.0f), f_param(n, 9, 32.0f), f_param(n, 10, 1.0f));
        }
        if (op == "GGML_OP_FLASH_ATTN_EXT") {
            // floats [0..2] = scale, max_bias, logit_softcap.
            ggml_tensor* r = ggml_flash_attn_ext(c, in(n, 0), in(n, 1), in(n, 2),
                                                 n["inputs"].size() > 3 ? in(n, 3) : nullptr,
                                                 f_param(n, 0, 1.0f), f_param(n, 1),
                                                 f_param(n, 2));
            // The dumped graph records FA's output type; llama.cpp sets it explicitly and the
            // default does not always match.
            ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
            return r;
        }
        return nullptr;
    }

    void build_nodes(const json& doc) {
        for (const auto& n : doc["nodes"]) {
            const std::string id = n["id"];
            const std::string op = n["op"];
            ggml_tensor* t = nullptr;

            // SET_ROWS also carries a view_src (its destination), but it is a real op, so the
            // view shortcut must not swallow it.
            if (n.contains("view_src") && op != "GGML_OP_SET_ROWS") {
                t = rebuild_view(n);
            } else {
                t = build_node(n);
            }
            if (!t) {
                im.out.unsupported.insert(op);
                continue;
            }
            const std::string name = n["name"];
            ggml_set_name(t, name.empty() ? id.c_str() : name.c_str());
            by_id[id] = t;
            im.out.map[id] = t;
            im.out.order.push_back(t);
            im.out.last = t;
        }
    }

    // llama.cpp leaves most inputs ggml-autonamed, so recover their identity from the op that
    // consumes them and publish canonical names. Callers then bind inputs the same way for both
    // graph sources. Verified against the artifact: ROPE[src1] x32 = positions, SET_ROWS[src1]
    // = cache row index, FLASH_ATTN_EXT[src3] = mask, GET_ROWS[src1] = tokens / out_ids.
    void assign_roles(const json& doc) {
        std::map<std::string, std::string> role;  // leaf id -> canonical name
        int n_setrows = 0, n_getrows = 0;
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
                // Two distinct index leaves exist (one per cache); both are the same value, so
                // number them and let the caller write whichever it finds.
                const std::string id = ins[1];
                if (by_id.count(id) && !role.count(id)) {
                    role[id] = n_setrows++ == 0 ? "inp_kv_idx" : "inp_kv_idx_1";
                }
            } else if (op == "GGML_OP_FLASH_ATTN_EXT") {
                tag(3, "self_kq_mask");
            } else if (op == "GGML_OP_GET_ROWS") {
                // The first GET_ROWS over the token embedding takes the tokens; a later one
                // selects which rows to emit logits for.
                const std::string id = ins[1];
                if (by_id.count(id) && !role.count(id)) {
                    role[id] = n_getrows++ == 0 ? "inp_tokens" : "inp_out_ids";
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

std::shared_ptr<GgmlModel> GgmlModel::from_cgraph(const std::string& cgraph_json,
                                                  const std::string& gguf_path,
                                                  const std::string& backend_name) {
    std::ifstream f(cgraph_json);
    OPENVINO_ASSERT(f.good(), "[GGML] cannot open cgraph artifact: ", cgraph_json);
    json doc;
    f >> doc;
    OPENVINO_ASSERT(doc.value("format", "") == "ov-cgraph-v1",
                    "[GGML] unrecognised cgraph artifact format: ", doc.value("format", "<none>"));

    std::shared_ptr<GgmlModel> model(new GgmlModel());
    auto& im = *model->m_impl;
    im.init_backend(backend_name);
    im.load_weights(gguf_path);
    im.init_contexts();

    im.n_kv = doc.value("n_kv", 0);

    Loader loader{im, {}};
    loader.build_leaves(doc);
    loader.build_nodes(doc);
    im.throw_if_unsupported();
    loader.assign_roles(doc);
    im.finalize();
    return model;
}

size_t GgmlModel::context_size() const {
    return m_impl->n_kv;
}

}  // namespace ggml_emitter
}  // namespace ov
