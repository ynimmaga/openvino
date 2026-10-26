// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The public GgmlModel surface shared by every graph source: backend/buffer plumbing, input and
// output access, weight/metadata read-back. The graph itself is produced elsewhere -- here by the
// cgraph loader (cgraph_loader.cpp), which replays a topology llama.cpp described offline.
//
// This file links ggml; consumers of the public header do not.

#include "openvino/ggml_cgraph_loader/ggml_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "model_impl.hpp"

namespace ov {
namespace ggml_cgraph_loader {

GgmlModel::GgmlModel() : m_impl(new Impl()) {}
GgmlModel::~GgmlModel() = default;

namespace {

// GGML_COMPARE_CPU=<step>: on that compute() call, run the graph node by node on the model's backend
// and on a CPU copy, and print the nodes whose results diverge most (debugging backend numerics).
struct CompareState {
    int printed = 0;
};

bool compare_node(int index, ggml_tensor* t1, ggml_tensor* t2, void* user_data) {
    auto* st = static_cast<CompareState*>(user_data);
    if (t1->type != GGML_TYPE_F32 || st->printed >= 12) {
        return true;
    }
    const size_t n = ggml_nelements(t1);
    std::vector<float> a(n), b(n);
    ggml_backend_tensor_get(t1, a.data(), 0, n * sizeof(float));
    ggml_backend_tensor_get(t2, b.data(), 0, n * sizeof(float));
    double max_diff = 0, max_ref = 0;
    for (size_t i = 0; i < n; i++) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
        max_ref = std::max(max_ref, static_cast<double>(std::fabs(b[i])));
    }
    const double rel = max_diff / (max_ref + 1e-6);
    if (rel > 1e-2 || std::isnan(max_diff)) {
        fprintf(stderr, "[GGML compare] node %d %s (%s): max |diff| %.4g, max |cpu| %.4g, rel %.3g\n", index,
                ggml_get_name(t1), ggml_op_desc(t1), max_diff, max_ref, rel);
        st->printed++;
    }
    return true;
}

}  // namespace

namespace {

// GGML_REF=<file> (from dump_cgraph REF_OUT=<file> NO_FILL=1): after the first compute(), compare every
// node against the value llama.cpp had for it and print the first nodes that diverge. Errors
// propagate, so the first one is the faulty op (or a non-contiguous view feeding it).
struct RefEntry {
    std::string name;
    int64_t ne[4];
    std::vector<float> values;
};

std::vector<RefEntry> load_ref(const char* path) {
    std::vector<RefEntry> out;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[GGML ref] cannot open %s\n", path);
        return out;
    }
    uint32_t len;
    while (fread(&len, sizeof(len), 1, f) == 1) {
        RefEntry e;
        e.name.resize(len);
        uint64_t count = 0;
        if (fread(e.name.data(), 1, len, f) != len || fread(e.ne, sizeof(int64_t), 4, f) != 4 ||
            fread(&count, sizeof(count), 1, f) != 1) {
            break;
        }
        e.values.resize(count);
        if (count && fread(e.values.data(), sizeof(float), count, f) != count) {
            break;
        }
        out.push_back(std::move(e));
    }
    fclose(f);
    return out;
}

void check_against_ref(const std::vector<ggml_tensor*>& order, const std::vector<RefEntry>& ref) {
    if (order.size() != ref.size()) {
        fprintf(stderr, "[GGML ref] node count differs: replay %zu vs reference %zu\n", order.size(), ref.size());
        return;
    }
    int shown = 0;
    size_t compared = 0;
    double last_cos = 1.0;
    std::string last_name;
    for (size_t i = 0; i < order.size(); i++) {
        if (shown >= 3 && i + 1 < order.size()) { /* keep scanning for the last node's similarity */ }
        ggml_tensor* t = order[i];
        const RefEntry& e = ref[i];
        // A cache holds whatever the other graph slots contain, and the replay keeps its tokens in
        // different slots than llama.cpp does; what is read back from it is checked downstream.
        const std::string node_name = ggml_get_name(t);
        if (node_name.rfind("cache_", 0) == 0 || node_name.find("state") != std::string::npos) {
            continue;  // updated in place later in the graph; compared downstream instead
        }
        if (e.values.empty() || !ggml_is_contiguous(t) || ggml_nelements(t) != static_cast<int64_t>(e.values.size()) ||
            (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32)) {
            continue;
        }
        std::vector<float> v(e.values.size());
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        } else if (t->type == GGML_TYPE_I32) {
            std::vector<int32_t> iv(v.size());
            ggml_backend_tensor_get(t, iv.data(), 0, iv.size() * sizeof(int32_t));
            for (size_t k = 0; k < v.size(); k++) v[k] = static_cast<float>(iv[k]);
        } else {
            std::vector<ggml_fp16_t> h(v.size());
            ggml_backend_tensor_get(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
            for (size_t k = 0; k < v.size(); k++) v[k] = ggml_fp16_to_fp32(h[k]);
        }
        compared++;
        {
            double dot = 0, na = 0, nb = 0;
            for (size_t k = 0; k < v.size(); k++) {
                dot += static_cast<double>(v[k]) * e.values[k];
                na += static_cast<double>(v[k]) * v[k];
                nb += static_cast<double>(e.values[k]) * e.values[k];
            }
            last_cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
            last_name = ggml_get_name(t);
        }
        static const int ref_show = std::getenv("GGML_REF_SHOW") ? std::atoi(std::getenv("GGML_REF_SHOW")) : 3;
        static const double ref_tol = std::getenv("GGML_REF_TOL") ? std::atof(std::getenv("GGML_REF_TOL")) : 0.03;
        double max_diff = 0, max_ref = 0;
        bool nan = false;
        for (size_t k = 0; k < v.size(); k++) {
            if (std::isnan(v[k])) nan = true;
            max_diff = std::max(max_diff, static_cast<double>(std::fabs(v[k] - e.values[k])));
            max_ref = std::max(max_ref, static_cast<double>(std::fabs(e.values[k])));
        }
        const double rel = max_diff / (max_ref + 1e-6);
        if ((nan || rel > ref_tol) && shown < ref_show) {
            fprintf(stderr, "[GGML ref] node %zu %s (%s) ne=[%lld,%lld,%lld,%lld]: max |diff| %.4g, max |ref| %.4g, rel %.3g\n",
                    i, ggml_get_name(t), ggml_op_desc(t), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                    (long long)t->ne[3], max_diff, max_ref, rel);
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
                const ggml_tensor* x = t->src[s];
                fprintf(stderr, "    src%d %s (%s) ne=[%lld,%lld,%lld,%lld] contiguous=%d\n", s, ggml_get_name(x),
                        ggml_op_desc(x), (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2],
                        (long long)x->ne[3], ggml_is_contiguous(x) ? 1 : 0);
            }
            shown++;
        }
    }
    fprintf(stderr, "[GGML ref] compared %zu nodes, %d diverging shown; last compared node '%s' cosine similarity %.6f\n", compared,
            shown, last_name.c_str(), last_cos);
}

}  // namespace

bool GgmlModel::compute() {
    static const std::vector<RefEntry> ref = std::getenv("GGML_REF") ? load_ref(std::getenv("GGML_REF")) : std::vector<RefEntry>{};
    static bool ref_done = false;
    if (!ref.empty() && !ref_done) {
        ref_done = true;
        const bool ok = ggml_backend_graph_compute(m_impl->backend, m_impl->gf) == GGML_STATUS_SUCCESS;
        check_against_ref(m_impl->out.order, ref);
        return ok;
    }
    static const int compare_step = std::getenv("GGML_COMPARE_CPU") ? std::atoi(std::getenv("GGML_COMPARE_CPU")) : -1;
    static int step = 0;
    if (compare_step >= 0 && step++ == compare_step && !m_impl->sched) {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        CompareState st;
        const bool ok = ggml_backend_compare_graph_backend(m_impl->backend, cpu, m_impl->gf, compare_node, &st, nullptr, 0);
        ggml_backend_free(cpu);
        fprintf(stderr, "[GGML compare] step %d done, %d diverging nodes printed\n", compare_step, st.printed);
        return ok;
    }
    if (m_impl->sched) {
        return ggml_backend_sched_graph_compute(m_impl->sched, m_impl->gf) == GGML_STATUS_SUCCESS;
    }
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

size_t GgmlModel::output_size() const {
    const ggml_tensor* t = m_impl->out.last;
    return static_cast<size_t>(t->ne[0]) * t->ne[1] * t->ne[2] * t->ne[3];
}

void GgmlModel::read_output(float* out) const {
    ggml_backend_tensor_get(m_impl->out.last, out, 0, output_size() * sizeof(float));
}

const std::map<std::string, ggml_tensor*>& GgmlModel::externals() const {
    return m_impl->out.externals;
}

size_t GgmlModel::input_nbytes(const std::string& name) const {
    auto it = m_impl->out.externals.find(name);
    return it == m_impl->out.externals.end() ? 0 : ggml_nbytes(it->second);
}

bool GgmlModel::read_input(const std::string& name, void* dst, size_t bytes) const {
    auto it = m_impl->out.externals.find(name);
    if (it == m_impl->out.externals.end() || bytes > ggml_nbytes(it->second)) {
        return false;
    }
    ggml_backend_tensor_get(it->second, dst, 0, bytes);
    return true;
}

size_t GgmlModel::weight_nbytes(const std::string& gguf_name) const {
    auto it = m_impl->wmap.find(gguf_name);
    return it == m_impl->wmap.end() ? 0 : ggml_nbytes(it->second);
}

bool GgmlModel::read_weight(const std::string& gguf_name, size_t offset_bytes, void* dst,
                            size_t nbytes) const {
    auto it = m_impl->wmap.find(gguf_name);
    if (it == m_impl->wmap.end() || offset_bytes + nbytes > ggml_nbytes(it->second)) {
        return false;
    }
    ggml_backend_tensor_get(it->second, dst, offset_bytes, nbytes);
    return true;
}

bool GgmlModel::gguf_meta_f32(const std::string& key, float& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0 || gguf_get_kv_type(m_impl->gg, id) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    out = gguf_get_val_f32(m_impl->gg, id);
    return true;
}

bool GgmlModel::gguf_meta_str(const std::string& key, std::string& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0 || gguf_get_kv_type(m_impl->gg, id) != GGUF_TYPE_STRING) {
        return false;
    }
    out = gguf_get_val_str(m_impl->gg, id);
    return true;
}

bool GgmlModel::gguf_meta_i32(const std::string& key, int32_t& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0) {
        return false;
    }
    // GGUF writers vary on the signedness of a plain integer field (e.g. image_size is UINT32
    // here); gguf_get_val_i32/u32 assert on an exact type match, so try both.
    switch (gguf_get_kv_type(m_impl->gg, id)) {
    case GGUF_TYPE_INT32:
        out = gguf_get_val_i32(m_impl->gg, id);
        return true;
    case GGUF_TYPE_UINT32:
        out = static_cast<int32_t>(gguf_get_val_u32(m_impl->gg, id));
        return true;
    default:
        return false;
    }
}

bool GgmlModel::gguf_meta_f32_array(const std::string& key, std::vector<float>& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0 || gguf_get_arr_type(m_impl->gg, id) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    const size_t n = gguf_get_arr_n(m_impl->gg, id);
    const float* data = static_cast<const float*>(gguf_get_arr_data(m_impl->gg, id));
    out.assign(data, data + n);
    return true;
}

namespace {

// Raw element data of a GGUF array of exactly `type`, or nullptr.
const void* gguf_array(gguf_context* gg, const std::string& key, gguf_type type, size_t& n) {
    if (!gg) {
        return nullptr;
    }
    const int64_t id = gguf_find_key(gg, key.c_str());
    if (id < 0 || gguf_get_kv_type(gg, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gg, id) != type) {
        return nullptr;
    }
    n = gguf_get_arr_n(gg, id);
    return gguf_get_arr_data(gg, id);
}

}  // namespace

bool GgmlModel::gguf_meta_i32_array(const std::string& key, std::vector<int32_t>& out) const {
    size_t n = 0;
    const auto* data = static_cast<const int32_t*>(gguf_array(m_impl->gg, key, GGUF_TYPE_INT32, n));
    if (!data) {
        return false;
    }
    out.assign(data, data + n);
    return true;
}

bool GgmlModel::gguf_meta_u64_array(const std::string& key, std::vector<uint64_t>& out) const {
    size_t n = 0;
    const auto* data = static_cast<const uint64_t*>(gguf_array(m_impl->gg, key, GGUF_TYPE_UINT64, n));
    if (!data) {
        return false;
    }
    out.assign(data, data + n);
    return true;
}

bool GgmlModel::gguf_meta_u8_array(const std::string& key, std::vector<uint8_t>& out) const {
    size_t n = 0;
    const auto* data = static_cast<const uint8_t*>(gguf_array(m_impl->gg, key, GGUF_TYPE_UINT8, n));
    if (!data) {
        return false;
    }
    out.assign(data, data + n);
    return true;
}

bool GgmlModel::gguf_meta_str_array(const std::string& key, std::vector<std::string>& out) const {
    if (!m_impl->gg) {
        return false;
    }
    const int64_t id = gguf_find_key(m_impl->gg, key.c_str());
    if (id < 0 || gguf_get_kv_type(m_impl->gg, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(m_impl->gg, id) != GGUF_TYPE_STRING) {
        return false;
    }
    const size_t n = gguf_get_arr_n(m_impl->gg, id);
    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(m_impl->gg, id, i));
    }
    return true;
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

bool GgmlModel::uses_alibi() const {
    return m_impl->alibi;
}

std::string GgmlModel::backend_name() const {
    std::string name = ggml_backend_name(m_impl->backend);
    if (!m_impl->fallback_ops.empty()) {
        name += " (CPU for";
        for (const auto& op : m_impl->fallback_ops) {
            name += " " + op;
        }
        name += ")";
    }
    return name;
}

std::vector<std::string> GgmlModel::output_names() const {
    std::vector<std::string> names;
    for (const auto& kv : m_impl->out.named_outputs) {
        names.push_back(kv.first);
    }
    return names;
}

size_t GgmlModel::named_output_size(const std::string& name) const {
    auto it = m_impl->out.named_outputs.find(name);
    return it == m_impl->out.named_outputs.end() ? 0 : ggml_nelements(it->second);
}

bool GgmlModel::read_named_output(const std::string& name, float* dst) const {
    auto it = m_impl->out.named_outputs.find(name);
    if (it == m_impl->out.named_outputs.end() || it->second->type != GGML_TYPE_F32) {
        return false;
    }
    ggml_backend_tensor_get(it->second, dst, 0, ggml_nbytes(it->second));
    return true;
}

bool GgmlModel::has_recurrent_state() const {
    if (!m_impl->out.state_names.empty()) {
        return true;
    }
    for (const auto& kv : m_impl->out.externals) {
        if (kv.first.rfind("cache_r_", 0) == 0 || kv.first.rfind("cache_s_", 0) == 0) {
            return true;
        }
    }
    return false;
}

void GgmlModel::zero_recurrent_state() {
    for (const auto& kv : m_impl->out.externals) {
        if (kv.first.rfind("cache_r_", 0) == 0 || kv.first.rfind("cache_s_", 0) == 0 || m_impl->out.state_names.count(kv.first)) {
            ggml_backend_tensor_memset(kv.second, 0, 0, ggml_nbytes(kv.second));
        }
    }
}

size_t GgmlModel::zero_caches(const std::string& prefix) {
    size_t n = 0;
    for (const auto& kv : m_impl->out.externals) {
        if (kv.first.rfind(prefix, 0) == 0) {
            ggml_backend_tensor_memset(kv.second, 0, 0, ggml_nbytes(kv.second));
            n++;
        }
    }
    return n;
}

void GgmlModel::copy_cache_slots(size_t src, size_t dst, size_t n) {
    std::vector<uint8_t> rows;
    for (const auto& kv : m_impl->out.externals) {
        ggml_tensor* t = kv.second;
        if (kv.first.rfind("cache_", 0) != 0 || static_cast<size_t>(t->ne[1]) < std::max(src, dst) + n) {
            continue;
        }
        const size_t row = t->nb[1];
        rows.resize(n * row);
        ggml_backend_tensor_get(t, rows.data(), src * row, rows.size());
        ggml_backend_tensor_set(t, rows.data(), dst * row, rows.size());
    }
}

}  // namespace ggml_cgraph_loader
}  // namespace ov
