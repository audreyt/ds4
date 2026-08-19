#include <metal_stdlib>
using namespace metal;

constant uint QWEN_GDN_V_HEADS = 48u;
constant uint QWEN_GDN_K_HEADS = 16u;
constant uint QWEN_GDN_HEAD_DIM = 128u;
constant uint QWEN_GDN_CONV_K = 4u;

struct ds4_qwen_gdn_conv_args {
    uint32_t n_channels;
    uint32_t layer;
};

struct ds4_qwen_gdn_core_args {
    uint32_t layer;
};

kernel void kernel_qwen_gdn_conv(
        constant ds4_qwen_gdn_conv_args & args,
        device const float * qkv,
        device const float * conv_w,
        device float * conv,
        device float * mixed,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_channels) return;
    device float *h = conv + ((uint)args.layer * args.n_channels + gid) * QWEN_GDN_CONV_K;
    const float x = qkv[gid];
    h[0] = h[1];
    h[1] = h[2];
    h[2] = h[3];
    h[3] = x;
    const device float *w = conv_w + gid * QWEN_GDN_CONV_K;
    float acc = w[0]*h[0] + w[1]*h[1] + w[2]*h[2] + w[3]*h[3];
    mixed[gid] = acc / (1.0f + exp(-acc));
}

kernel void kernel_qwen_gdn_core(
        constant ds4_qwen_gdn_core_args & args,
        device const float * mixed,
        device const float * z,
        device const float * alpha,
        device const float * beta,
        device const float * A_log,
        device const float * dt_bias,
        device const float * snorm,
        device float * state,
        device float * core,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  lid   [[thread_index_in_threadgroup]]) {
    const uint vh = tgpig.x;
    const uint j = lid;
    if (vh >= QWEN_GDN_V_HEADS || j >= QWEN_GDN_HEAD_DIM) return;

    const uint kh = vh % QWEN_GDN_K_HEADS;
    threadgroup float tq[QWEN_GDN_HEAD_DIM];
    threadgroup float tk[QWEN_GDN_HEAD_DIM];
    threadgroup float toh[QWEN_GDN_HEAD_DIM];

    const device float *qsrc = mixed + kh * QWEN_GDN_HEAD_DIM;
    const device float *ksrc = mixed + 2048u + kh * QWEN_GDN_HEAD_DIM;
    tq[j] = qsrc[j];
    tk[j] = ksrc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qn = tq[j] * tq[j];
    float kn = tk[j] * tk[j];
    qn += simd_shuffle_xor(qn, 1u);
    qn += simd_shuffle_xor(qn, 2u);
    qn += simd_shuffle_xor(qn, 4u);
    qn += simd_shuffle_xor(qn, 8u);
    qn += simd_shuffle_xor(qn, 16u);
    kn += simd_shuffle_xor(kn, 1u);
    kn += simd_shuffle_xor(kn, 2u);
    kn += simd_shuffle_xor(kn, 4u);
    kn += simd_shuffle_xor(kn, 8u);
    kn += simd_shuffle_xor(kn, 16u);
    threadgroup float qn_sg[4];
    threadgroup float kn_sg[4];
    const uint sg = lid / 32u;
    if ((lid & 31u) == 0u) {
        qn_sg[sg] = qn;
        kn_sg[sg] = kn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qn = qn_sg[0] + qn_sg[1] + qn_sg[2] + qn_sg[3];
    kn = kn_sg[0] + kn_sg[1] + kn_sg[2] + kn_sg[3];
    qn = 1.0f / max(sqrt(qn), 1e-6f);
    kn = 1.0f / max(sqrt(kn), 1e-6f);
    tq[j] *= qn;
    tk[j] *= kn;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float vj = mixed[4096u + vh * QWEN_GDN_HEAD_DIM + j];
    float g = alpha[vh] + dt_bias[vh];
    if (g > 20.0f) {
    } else if (g < -20.0f) {
        g = exp(g);
    } else {
        g = log(1.0f + exp(g));
    }
    const float decay = exp(g * A_log[vh]);
    const float b = 1.0f / (1.0f + exp(-beta[vh]));

    device float *Sj = state + ((uint)args.layer * QWEN_GDN_V_HEADS + vh) *
                       QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM + j * QWEN_GDN_HEAD_DIM;
    float kv = 0.0f;
    for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u) {
        float4 s = *(device float4 *)(Sj + i);
        s *= decay;
        kv += s.x * tk[i] + s.y * tk[i + 1] + s.z * tk[i + 2] + s.w * tk[i + 3];
        *(device float4 *)(Sj + i) = s;
    }
    const float delta = (vj - kv) * b;
    float oh = 0.0f;
    const float qscale = 1.0f / sqrt((float)QWEN_GDN_HEAD_DIM);
    for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u) {
        float4 s = *(device float4 *)(Sj + i);
        s.x += tk[i] * delta;
        s.y += tk[i + 1] * delta;
        s.z += tk[i + 2] * delta;
        s.w += tk[i + 3] * delta;
        *(device float4 *)(Sj + i) = s;
        oh += s.x * tq[i] + s.y * tq[i + 1] + s.z * tq[i + 2] + s.w * tq[i + 3];
    }
    oh *= qscale;
    toh[j] = oh;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ss = toh[j] * toh[j];
    ss += simd_shuffle_xor(ss, 1u);
    ss += simd_shuffle_xor(ss, 2u);
    ss += simd_shuffle_xor(ss, 4u);
    ss += simd_shuffle_xor(ss, 8u);
    ss += simd_shuffle_xor(ss, 16u);
    threadgroup float ss_sg[4];
    if ((lid & 31u) == 0u) ss_sg[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = (ss_sg[0] + ss_sg[1] + ss_sg[2] + ss_sg[3]) / (float)QWEN_GDN_HEAD_DIM;
    const float nscale = 1.0f / sqrt(ss + 1e-6f);
    const float zj = z[vh * QWEN_GDN_HEAD_DIM + j];
    core[vh * QWEN_GDN_HEAD_DIM + j] = oh * nscale * snorm[j] * (zj / (1.0f + exp(-zj)));
}

struct ds4_qwen_split_q_args {
    uint32_t n_head;
    uint32_t head_dim;
};

kernel void kernel_qwen_split_gated_q(
        constant ds4_qwen_split_q_args & args,
        device const float * q_raw,
        device float * q,
        device float * gate,
        uint gid [[thread_position_in_grid]]) {
    const uint n = args.n_head * args.head_dim;
    if (gid >= n) return;
    const uint h = gid / args.head_dim;
    const uint d = gid % args.head_dim;
    q[gid] = q_raw[h * args.head_dim * 2u + d];
    gate[gid] = q_raw[h * args.head_dim * 2u + args.head_dim + d];
}

struct ds4_qwen_rope_args {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos;
    float freq_base;
};

kernel void kernel_qwen_rope_rotate_half(
        constant ds4_qwen_rope_args & args,
        device float * x,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint n_half = args.n_rot / 2u;
    if (h >= args.n_head || lid >= n_half) return;
    const float theta_scale = pow(args.freq_base, -2.0f / (float)args.n_rot);
    float theta = (float)args.pos;
    for (uint i = 0; i < lid; i++) theta *= theta_scale;
    device float *xh = x + h * args.head_dim;
    const float c = cos(theta);
    const float s = sin(theta);
    const float x0 = xh[lid];
    const float x1 = xh[lid + n_half];
    xh[lid] = x0 * c - x1 * s;
    xh[lid + n_half] = x0 * s + x1 * c;
}

struct ds4_qwen_fullattn_args {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t pos;
    uint32_t layer;
    uint32_t cap;
    uint32_t gated;
};

kernel void kernel_qwen_attn_decode(
        constant ds4_qwen_fullattn_args & args,
        device const float * q,
        device const float * k_cache,
        device const float * v_cache,
        device const float * gate,
        device float * heads,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint d = lid;
    if (h >= args.n_head || d >= args.head_dim) return;
    const uint kv_dim = args.n_head_kv * args.head_dim;
    const uint kv_h = h / (args.n_head / args.n_head_kv);
    const device float *base_k = k_cache + ((uint)args.layer * args.cap) * kv_dim;
    const device float *base_v = v_cache + ((uint)args.layer * args.cap) * kv_dim;
    threadgroup float qh[256];
    qh[d] = q[h * args.head_dim + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = 1.0f / sqrt((float)args.head_dim);
    float m_prev = -1.0e30f;
    float l_prev = 0.0f;
    float acc = 0.0f;
    const uint sg = lid / 32u;
    for (uint t = 0; t <= args.pos; t++) {
        const device float *kt = base_k + t * kv_dim + kv_h * args.head_dim;
        float partial = qh[d] * kt[d];
        partial += simd_shuffle_xor(partial, 1u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 8u);
        partial += simd_shuffle_xor(partial, 16u);
        threadgroup float dot_sg[8];
        if ((lid & 31u) == 0u) dot_sg[sg] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (dot_sg[0] + dot_sg[1] + dot_sg[2] + dot_sg[3]
                     + dot_sg[4] + dot_sg[5] + dot_sg[6] + dot_sg[7]) * scale;
        const float m_curr = max(m_prev, score);
        const float al = exp(m_prev - m_curr);
        const float be = exp(score - m_curr);
        l_prev = l_prev * al + be;
        const device float *vt = base_v + t * kv_dim + kv_h * args.head_dim;
        acc = acc * al + be * vt[d];
        m_prev = m_curr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float out = acc / (l_prev > 1e-8f ? l_prev : 1.0f);
    if (args.gated) {
        const float g = gate[h * args.head_dim + d];
        out *= 1.0f / (1.0f + exp(-g));
    }

    heads[h * args.head_dim + d] = out;
}

struct ds4_qwen_q6k_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_bytes;
};

kernel void kernel_qwen_q6k_matvec(
        constant ds4_qwen_q6k_args & args,
        device const char * w,
        device const float * x,
        device float * out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.out_dim) return;
    device const char *row = w + (uint64_t)gid * args.row_bytes;
    const int nb = (int)args.in_dim / 256;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        device const uchar *ql = (device const uchar *)(row + (uint)i * 210u);
        device const uchar *qh = ql + 128;
        device const char *scales = (device const char *)(qh + 64);
        const float d = float(*((device const half *)(scales + 16)));
        device const float *yb = x + (uint)i * 256;
        for (int n128 = 0; n128 < 256; n128 += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = ((int)(ql[l + 0]  & 0x0F) | (((int)qh[l] << 4) & 0x30)) - 32;
                const int q2 = ((int)(ql[l + 32] & 0x0F) | (((int)qh[l] << 2) & 0x30)) - 32;
                const int q3 = ((int)(ql[l + 0]  >> 4)    | (((int)qh[l] << 0) & 0x30)) - 32;
                const int q4 = ((int)(ql[l + 32] >> 4)    | (((int)qh[l] >> 2) & 0x30)) - 32;
                sumf += d * (float)scales[is + 0] * (float)q1 * yb[n128 + l + 0];
                sumf += d * (float)scales[is + 2] * (float)q2 * yb[n128 + l + 32];
                sumf += d * (float)scales[is + 4] * (float)q3 * yb[n128 + l + 64];
                sumf += d * (float)scales[is + 6] * (float)q4 * yb[n128 + l + 96];
            }
            ql += 64;
            qh += 32;
            scales += 8;
        }
    }
    out[gid] = sumf;
}

kernel void kernel_qwen_copy_f32(
        device const float * src,
        device float * dst,
        constant uint & n,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = src[gid];
}

