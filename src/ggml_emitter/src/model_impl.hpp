// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Internal: the pieces both graph sources share.
//
// A GgmlModel can be built two ways -- from OpenVINO's GGUF decoder builder (emitter.cpp) or
// from a dumped llama.cpp cgraph artifact (cgraph_loader.cpp). Everything AROUND the graph is
// identical: backend selection, weight upload, the external context that keeps inputs and KV
// caches alive across compute() calls, graph allocation. Only the middle step differs.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include "openvino/core/except.hpp"
#include "openvino/ggml_emitter/emitter.hpp"

namespace ov {
namespace ggml_emitter {

constexpr int MAX_NODES = 8192;

/// Graph outputs, owned by the caller: with the emitter path the producer is destroyed together
/// with the DecoderBuilder, so anything needed afterwards must live here.
struct Out {
    std::map<std::string, ggml_tensor*> map, externals;
    std::set<std::string> unsupported;
    ggml_tensor* last = nullptr;
    /// Every node, in topological order. Needed because a forward expansion from the logits
    /// alone MISSES the KV cache writes: SET_ROWS results are not reachable from the output, so
    /// they would be dropped from the graph and attention would read an empty cache. Expanding
    /// in this order also reproduces the original node ordering, which is what makes the cache
    /// write land before the read.
    std::vector<ggml_tensor*> order;
};

struct GgmlModel::Impl {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, ebuf = nullptr;
    ggml_context *wctx = nullptr, *ctx_g = nullptr, *ctx_ext = nullptr;
    gguf_context* gg = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    std::map<std::string, ggml_tensor*> wmap;  // gguf tensor name -> uploaded weight
    size_t n_kv = 0;  // baked into the graph shapes; 0 when unknown (emitter path sets it too)
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

    /// Pick and initialise a ggml backend. Empty name means GPU, then IGPU (Intel integrated
    /// graphics register as IGPU, not GPU), then CPU.
    void init_backend(const std::string& backend_name) {
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
        backend = ggml_backend_dev_init(dev, nullptr);
        OPENVINO_ASSERT(backend, "[GGML] failed to initialise backend");
    }

    /// Read every tensor out of the .gguf and upload it to the backend, keyed by gguf name.
    void load_weights(const std::string& gguf_path) {
        gguf_init_params gp = {true, &wctx};
        gg = gguf_init_from_file(gguf_path.c_str(), gp);
        OPENVINO_ASSERT(gg, "[GGML] cannot read gguf: ", gguf_path);
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        FILE* f = fopen(gguf_path.c_str(), "rb");
        OPENVINO_ASSERT(f, "[GGML] cannot open: ", gguf_path);
        const size_t doff = gguf_get_data_offset(gg);
        std::vector<uint8_t> tmp;
        for (ggml_tensor* t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            const int i = gguf_find_tensor(gg, ggml_get_name(t));
            if (i < 0) {
                continue;
            }
            tmp.resize(ggml_nbytes(t));
            if (fseek(f, static_cast<long>(doff + gguf_get_tensor_offset(gg, i)), SEEK_SET) != 0 ||
                fread(tmp.data(), 1, tmp.size(), f) != tmp.size()) {
                fclose(f);
                OPENVINO_THROW("[GGML] short read for tensor ", ggml_get_name(t));
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
            wmap[ggml_get_name(t)] = t;
        }
        fclose(f);
    }

    void init_contexts() {
        ctx_ext = ggml_init({ggml_tensor_overhead() * 1024, nullptr, true});
        ctx_g = ggml_init({ggml_tensor_overhead() * MAX_NODES +
                               ggml_graph_overhead_custom(MAX_NODES, false),
                           nullptr, true});
    }

    /// Allocate the external tensors, expand the forward graph, allocate it, and zero the KV
    /// caches. Shared tail of both build paths.
    void finalize() {
        OPENVINO_ASSERT(out.last, "[GGML] no output tensor produced");
        ebuf = ggml_backend_alloc_ctx_tensors(ctx_ext, backend);
        gf = ggml_new_graph_custom(ctx_g, MAX_NODES, false);
        if (out.order.empty()) {
            ggml_build_forward_expand(gf, out.last);
        } else {
            for (ggml_tensor* t : out.order) {
                ggml_build_forward_expand(gf, t);  // already-visited nodes are skipped
            }
        }
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        OPENVINO_ASSERT(ggml_gallocr_alloc_graph(alloc, gf), "[GGML] graph allocation failed");

        // KV caches live in ctx_ext, so they are allocated once and persist across compute()
        // calls; SET_ROWS writes through a view into that same buffer.
        for (const auto& kv : out.externals) {
            if (kv.first.rfind("cache_", 0) != 0) {
                continue;
            }
            std::vector<char> z(ggml_nbytes(kv.second), 0);
            ggml_backend_tensor_set(kv.second, z.data(), 0, z.size());
        }
    }

    void throw_if_unsupported() const {
        if (out.unsupported.empty()) {
            return;
        }
        std::string list;
        for (const auto& u : out.unsupported) {
            list += (list.empty() ? "" : ", ") + u;
        }
        OPENVINO_THROW("[GGML] cannot translate: ", list);
    }
};

}  // namespace ggml_emitter
}  // namespace ov
