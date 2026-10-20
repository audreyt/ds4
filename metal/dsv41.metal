// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_router {
    uint n_expert, top_k, has_bias;
    float scale;
};

// One-token router for up to 1024 experts in one dispatch: the softplus/sqrt
// probability transform, the biased top-k through the same bitonic network as
// kernel_argsort_f32_i32_desc (padding indices sort last), and the weights
// normalised as the generic chain does: gather, SIMD sum over the top_k lanes,
// clamp, divide, scale. Threadgroup memory holds nth ints and 2 * nth floats.
kernel void kernel_dsv41_router_one(
        constant ds4_metal_args_dsv41_router &args,
        device const float *logits,
        device const float *bias,
        device float *probs,
        device int32_t *selected,
        device float *weights,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 tpos [[thread_position_in_threadgroup]],
        uint2 tsize [[threads_per_threadgroup]]) {
    const uint col = tpos.x, ntg = tsize.x;
    const int width = (int)args.n_expert;
    logits += tg.y * args.n_expert;   /* one threadgroup per row */
    probs += tg.y * args.n_expert;
    selected += tg.y * args.top_k;
    weights += tg.y * args.top_k;
    threadgroup float *score = (threadgroup float *)(shmem_i32 + ntg);
    threadgroup float *prob = score + ntg;
    shmem_i32[col] = (int)col;
    if ((int)col < width) {
        const float x = logits[col];
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(sp);
        probs[col] = p;
        prob[col] = p;
        score[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = (int)col ^ j;
            if (ixj > (int)col) {
                const int a = shmem_i32[col], b = shmem_i32[ixj];
                const bool swap = ((int)col & k) == 0 ?
                    (a >= width || (b < width && score[a] < score[b])) :
                    (b >= width || (a < width && score[a] > score[b]));
                if (swap) { shmem_i32[col] = b; shmem_i32[ixj] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < args.top_k) {
        const int32_t idx = shmem_i32[col];
        selected[col] = idx;
        const float w = prob[idx];
        const float total = clamp(simd_sum(w), 6.103515625e-5f, INFINITY);
        // two stores as in the generic chain: the quotient is rounded before the scale
        threadgroup volatile float *quotient = (threadgroup volatile float *)(prob + ntg) + col;
        *quotient = w / total;
        weights[col] = *quotient * args.scale;
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
    uint round_all;
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong head = ((ulong)group.y * args.heads + group.x) * args.width;
    if (args.round_all) {
        for (uint k = lane; k + 64u < args.width; k += 32u) x[head + k] = dsv41_bf16(x[head + k]);
    }
    const ulong i = head + args.width - 64u + 2u * lane;
    float re = x[i], im = x[i + 1u];
    if (args.round_all) { re = dsv41_bf16(re); im = dsv41_bf16(im); }
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_rope_quantize {
    uint width, start, inverse, mode;
    float frequencies[32];
};

// One row's rope, quantization and store into a cache row: the rotation of
// the last 64 columns as kernel_dsv41_rope applies it, then the block
// quantization of kernel_dsv41_quantize, written to dst.
kernel void kernel_dsv41_rope_quantize(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? x[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float theta = float(args.start) * args.frequencies[(column - rotary) >> 1u];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        // the contraction kernel_dsv41_rope compiles to
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) dst[column] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_norm_pair {
    uint n0, n1, ntg0, ntg1;
    float eps;
};

// Two weighted RMS norms in one dispatch, each row on its own thread count
// (ntg0 threads then ntg1), so every reduction matches the single-row kernel.
kernel void kernel_dsv41_norm_pair(
        constant ds4_metal_args_dsv41_norm_pair &args,
        device const float4 *x0,
        device const float4 *w0,
        device float4 *y0,
        device const float4 *x1,
        device const float4 *w1,
        device float4 *y1,
        threadgroup float *shmem [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort2 tpos [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort tid = tpos.x;
    const bool second = tid >= args.ntg0;
    const uint ntg = second ? args.ntg1 : args.ntg0;
    const uint t = second ? tid - args.ntg0 : tid;
    const uint sg = second ? sgitg - args.ntg0 / 32u : sgitg;
    const int n = int(second ? args.n1 : args.n0);
    const int ne00_t = n / 4;
    const uint row = tg.y;   /* grid row = activation row */
    device const float4 *x = (second ? x1 : x0) + (ulong)row * ne00_t;
    device const float4 *w = second ? w1 : w0;
    device float4 *y = (second ? y1 : y0) + (ulong)row * ne00_t;
    threadgroup float *sh = shmem + (second ? 32u : 0u);
    if (sg == 0) {
        sh[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        sumf += dot(x[i], x[i]);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sh[sg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sh[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf/n;
    const float scale = 1.0f/sqrt(mean + args.eps);
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        float4 v = x[i]*scale;
        v = v*w[i];
        y[i] = ds4_bf16_round(v);
    }
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_indexer_all {
    uint rows;
    uint start;
    uint ratio;
    uint top_k;
};

// Rows with at most top_k visible keys attend to all of them.
kernel void kernel_dsv41_indexer_all(
        constant ds4_metal_args_dsv41_indexer_all &args,
        device int *ids,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.top_k || index.y >= args.rows) return;
    const uint visible = (args.start + index.y + 1u) / args.ratio;
    if (index.x < min(visible, args.top_k))
        ids[(ulong)index.y * args.top_k + index.x] = (int)index.x;
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
    uint mask_stride;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * args.mask_stride + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

/* The 2048 candidate blocks of a 0/-inf mask in ascending order; one
 * threadgroup of 1024 threads, four mask entries per thread and step. */
kernel void kernel_dsv41_candidate_list(
        constant uint &blocks,
        device const float *mask,
        device uint *list,
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[33];
    uint base = 0u;
    for (uint b0 = 0u; b0 < blocks; b0 += 4096u) {
        const uint b = b0 + tid * 4u;
        bool keep[4];
        uint count = 0u;
        for (uint j = 0u; j < 4u; j++) {
            keep[j] = b + j < blocks && mask[b + j] == 0.0f;
            count += keep[j] ? 1u : 0u;
        }
        const uint before = simd_prefix_exclusive_sum(count);
        if (lane == 31u) part[simd] = before + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const uint c = part[lane], c_before = simd_prefix_exclusive_sum(c);
            part[lane] = c_before;
            if (lane == 31u) part[32] = c_before + c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint at = base + part[simd] + before;
        for (uint j = 0u; j < 4u; j++) {
            if (keep[j] && at < 2048u) list[at] = b + j;
            at += keep[j] ? 1u : 0u;
        }
        base += part[32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Positions in the candidate list back to key ids, in place and in order. */
kernel void kernel_dsv41_candidate_ids(
        constant uint &count,
        device const uint *list,
        device int *selected,
        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint p = (uint)selected[i];
    selected[i] = (int)(list[p >> 3] * 8u + (p & 7u));
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif
