struct ds4_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. DS4 uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the DS4 graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;

// Q8_0 -> F32 dequantizing 1D copy.  Used by the compressor APE path so
// stock-recipe GGUFs that ship `*.compressor_ape.weight` as Q8_0 can be
// read through the same byte-strided copy that the F16/F32 ape paths use.
// args.ne00 is the total element count (must be divisible by QK8_0 = 32);
// src is a packed Q8_0 region (sizeof(block_q8_0) = 34 bytes per QK8_0 elements)
// and dst is contiguous F32.
//
// Uses explicit byte arithmetic instead of `block_q8_0 *` indexing because
// the GGUF byte stride (34) does not match Metal's natural struct alignment
// for `block_q8_0` (which would be padded to a multiple of `alignof(half)`
// in some cases).  Each thread handles one output element and re-reads the
// half scale from its block's first two bytes; that's redundant but cheap
// and the compressor APE total element count is tiny (a few thousand).
kernel void kernel_cpy_q8_0_f32(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int n = (int) args.ne00;
    const int gid = (int)(tgpig.x * ntg.x + tiitg);
    if (gid >= n) return;

    constexpr int BLOCK_BYTES = 34;
    const int blk = gid / QK8_0;
    const int idx = gid - blk * QK8_0;
    device const char *bp = src0 + (uint64_t)blk * BLOCK_BYTES;
    half d_h;
    /* half scale lives at the first 2 bytes of the block */
    thread char *dp = (thread char *) &d_h;
    dp[0] = bp[0]; dp[1] = bp[1];
    const float d = (float) d_h;
    const int8_t q = (int8_t) bp[2 + idx];
    device float *out = (device float *) dst;
    out[gid] = (float) q * d;
}
