// SPDX-License-Identifier: MIT
// Fixed-byte GGUF fixtures against the scalar ggml-quants.c dequantizers
// (llama.cpp 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a).
#include "../ds4_mmq.h"
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_C
#include "../ggml-common.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* As in tests/test_mmvq.cu: this isolated MMQ test has no Q8_0 activation
 * producer cache. None of the Q1 paths under test consumes that cache. */
extern "C" int ds4_cuda_q8_fold_take_q81(const void *, uint64_t, const void **q81) {
    if (q81) *q81 = nullptr;
    return 0;
}

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
    exit(1); } } while (0)
#define REQUIRE(call) do { if (!(call)) { fprintf(stderr, "%s:%d %s\n", \
    __FILE__, __LINE__, #call); exit(1); } } while (0)

static float half_float(uint16_t h) {
    __half_raw r; r.x = h;
    return __half2float(__half(r));
}

static void scale_min(int j, const uint8_t *s, uint8_t *scale, uint8_t *min) {
    if (j < 4) { *scale = s[j] & 63; *min = s[j + 4] & 63; }
    else {
        *scale = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
        *min = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

static void dequant(int type, const void *raw, float *y) {
    if (type == 13) {
        const block_q5_K &x = *(const block_q5_K *)raw;
        const float d = half_float(x.data.data.d), min = half_float(x.data.data.dmin);
        const uint8_t *ql = x.qs, *qh = x.qh;
        int is = 0; uint8_t sc, m, u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            scale_min(is, x.scales, &sc, &m);
            float d1 = d * sc, m1 = min * m;
            scale_min(is + 1, x.scales, &sc, &m);
            float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 15) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    } else if (type == 14) {
        const block_q6_K &x = *(const block_q6_K *)raw;
        const float d = half_float(x.d);
        const uint8_t *ql = x.ql, *qh = x.qh;
        const int8_t *sc = x.scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = ((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    } else {
        const block_iq1_s &x = *(const block_iq1_s *)raw;
        const float d = half_float(x.d);
        for (int ib = 0; ib < 8; ++ib) {
            float dl = d * (2 * ((x.qh[ib] >> 12) & 7) + 1);
            float delta = x.qh[ib] & 0x8000 ? -0.125f : 0.125f;
            for (int l = 0; l < 4; ++l) {
                const int8_t *grid = (const int8_t *)(iq1s_grid +
                    (x.qs[4 * ib + l] | (((x.qh[ib] >> (3 * l)) & 7) << 8)));
                for (int j = 0; j < 8; ++j) *y++ = dl * (grid[j] + delta);
            }
        }
    }
}

static uint32_t seed = 0x1234abcd;
static uint8_t byte() {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return seed & 255;
}

static void check(const char *tag, int type, int N, const std::vector<float> &got,
                  const std::vector<double> &ref) {
    double err = 0, norm = 0, maxerr = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        REQUIRE(std::isfinite(got[i]));
        double d = got[i] - ref[i]; err += d * d; norm += ref[i] * ref[i];
        maxerr = fmax(maxerr, fabs(d));
    }
    double relative = sqrt(err / fmax(norm, 1e-30));
    printf("%s type=%d N=%d relative_rms=%.9g max_abs=%.9g\n", tag, type, N, relative, maxerr);
    REQUIRE(relative < 3e-4);
}

static void check_cache_chunks() {
    /* A sparse global SSD cache whose uncompressed conservative stream-K
     * index exceeds 2^30. High selected ids span the old chunk boundary;
     * compaction must preserve their distinct packed weights and outputs. */
    const int E = 13159, M = 37, K = 5120, N = 16384, slots = 2;
    const int selected[] = {0, 13106, 13107, 13158};
    const size_t expert_bytes = (size_t)M * (K / 256) * sizeof(block_iq1_s);
    std::vector<uint8_t> packed((size_t)E * expert_bytes);
    std::vector<float> w((size_t)4 * M * K);
    for (int i = 0; i < 4; i++) {
        uint8_t *expert = packed.data() + selected[i] * expert_bytes;
        for (size_t j = 0; j < expert_bytes; j++) expert[j] = byte();
        for (int b = 0; b < M * (K / 256); b++) {
            const uint16_t d = 0x2800;
            memcpy(expert + b * sizeof(block_iq1_s), &d, 2);
            dequant(19, expert + b * sizeof(block_iq1_s),
                w.data() + (size_t)i * M * K + b * 256);
        }
    }
    std::vector<float> x((size_t)N * K);
    for (size_t i = 0; i < x.size(); i++) {
        /* Q8_1 also stores the block sum in half. Balanced integer pairs
         * keep both d=1 and sum=0 exact, isolating expert compaction from
         * activation rounding just like the small arithmetic fixtures. */
        const int lane = i % 32;
        const int value = lane < 2 ? 127 : lane / 2;
        x[i] = lane & 1 ? -value : value;
    }
    std::vector<int32_t> ids((size_t)N * slots);
    for (size_t i = 0; i < ids.size(); i++) ids[i] = selected[i % 4];
    std::vector<double> rows((size_t)4 * M);
    for (int e = 0; e < 4; e++) for (int r = 0; r < M; r++)
        for (int k = 0; k < K; k++)
            rows[e * M + r] += (double)w[((size_t)e * M + r) * K + k] * x[k];
    void *dw; float *dx, *dy; int32_t *di;
    CUDA(cudaMalloc(&dw, packed.size()));
    CUDA(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA(cudaMalloc(&di, ids.size() * sizeof(int32_t)));
    CUDA(cudaMalloc(&dy, ids.size() * M * sizeof(float)));
    CUDA(cudaMemcpy(dw, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(di, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    REQUIRE(ds4_mmq_iq1_s_moe(dw, dx, di, dy, M, K, N, E, slots, 0) == 0);
    std::vector<float> got(ids.size() * M);
    std::vector<double> ref(got.size());
    CUDA(cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < ids.size(); i++) for (int r = 0; r < M; r++)
        ref[i * M + r] = rows[(i % 4) * M + r];
    check("sparse-cache compaction", 19, N, got, ref);
    CUDA(cudaFree(dw)); CUDA(cudaFree(dx)); CUDA(cudaFree(di)); CUDA(cudaFree(dy));
}

int main() {
    REQUIRE(ds4_mmq_init(0) == 0);
    for (int type : {13, 14, 19}) for (int K : {512, 5120}) {
        const int M = 37, E = 4, used = 2;
        int bytes = type == 13 ? 176 : type == 14 ? 210 : 50;
        std::vector<uint8_t> packed((size_t)E * M * (K / 256) * bytes);
        for (auto &b : packed) b = byte();
        for (size_t off = 0; off < packed.size(); off += bytes) {
            uint16_t d = 0x2800; // exactly 1/32
            memcpy(packed.data() + off + (type == 14 ? 208 : 0), &d, 2);
            if (type == 13) memcpy(packed.data() + off + 2, &d, 2);
        }
        std::vector<float> w((size_t)E * M * K);
        for (size_t b = 0; b < packed.size() / bytes; ++b)
            dequant(type, packed.data() + b * bytes, w.data() + b * 256);
        void *dw; CUDA(cudaMalloc(&dw, packed.size()));
        CUDA(cudaMemcpy(dw, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        for (int N : {1, 4, 17}) {
            std::vector<float> x((size_t)N * K);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = i % 32 == 0 ? 127 : (int)(byte() % 255) - 127;
            float *dx, *dy;
            CUDA(cudaMalloc(&dx, x.size() * 4));
            CUDA(cudaMalloc(&dy, (size_t)N * used * M * 4));
            CUDA(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            REQUIRE(ds4_mmq_q1_dense(type, dw, dx, dy, M, N, K, 0) == 0);
            std::vector<float> got((size_t)N * M);
            CUDA(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<double> ref(got.size());
            for (int t = 0; t < N; ++t) for (int r = 0; r < M; ++r)
                for (int k = 0; k < K; ++k) ref[t * M + r] += (double)w[r * K + k] * x[t * K + k];
            check("dense", type, N, got, ref);
            if (type == 19) {
                for (int slots : {1, used}) {
                    std::vector<int32_t> ids(N * slots);
                    for (int i = 0; i < N * slots; ++i) ids[i] = (i * 3 + 1) % E;
                    int32_t *di; CUDA(cudaMalloc(&di, ids.size() * 4));
                    CUDA(cudaMemcpy(di, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                    REQUIRE(ds4_mmq_iq1_s_moe(dw, dx, di, dy, M, K, N, E, slots, 0) == 0);
                    got.resize((size_t)N * slots * M); ref.assign(got.size(), 0);
                    CUDA(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
                    for (int t = 0; t < N; ++t) for (int s = 0; s < slots; ++s)
                        for (int r = 0; r < M; ++r) for (int k = 0; k < K; ++k)
                            ref[(t * slots + s) * M + r] +=
                                (double)w[((ids[t * slots + s] * M + r) * K) + k] * x[t * K + k];
                    check("routed", type, N, got, ref);
                    CUDA(cudaFree(di));
                }
            }
            if (type == 14) {
                REQUIRE(ds4_mmq_q6_K_embed(dw, NULL, dx, 3, 1, K, 0) == 0);
                got.resize(K); ref.resize(K);
                CUDA(cudaMemcpy(got.data(), dx, K * 4, cudaMemcpyDeviceToHost));
                for (int k = 0; k < K; ++k) ref[k] = w[3 * K + k];
                check("embedding", type, N, got, ref);
                for (int k = 0; k < K; ++k) REQUIRE(got[k] == ref[k]);
                const int count = N < 4 ? N : 4;
                std::vector<int32_t> tokens(count);
                for (int t = 0; t < count; ++t) tokens[t] = 3 + 5 * t;
                int32_t *dt; CUDA(cudaMalloc(&dt, count * 4));
                CUDA(cudaMemcpy(dt, tokens.data(), count * 4, cudaMemcpyHostToDevice));
                REQUIRE(ds4_mmq_q6_K_embed(dw, dt, dx, 0, count, K, 0) == 0);
                got.resize((size_t)count * K); ref.resize(got.size());
                CUDA(cudaMemcpy(got.data(), dx, got.size() * 4, cudaMemcpyDeviceToHost));
                for (int t = 0; t < count; ++t) for (int k = 0; k < K; ++k)
                    ref[t * K + k] = w[tokens[t] * K + k];
                check("embedding batch", type, count, got, ref);
                for (size_t k = 0; k < got.size(); ++k) REQUIRE(got[k] == ref[k]);
                CUDA(cudaFree(dt));
            }
            CUDA(cudaFree(dx)); CUDA(cudaFree(dy));
        }
        CUDA(cudaFree(dw));
    }
    check_cache_chunks();
    puts("Q1 CUDA parity PASS");
}
