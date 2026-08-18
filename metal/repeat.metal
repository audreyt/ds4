// DS4 Metal repeat kernel used for HC embedding expansion.

struct ds4_metal_args_repeat {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Repeats a source row into the HC channel dimension. DS4 uses this when the
// token embedding has to become an HC activation block before layer 0.
template<typename T>
kernel void kernel_repeat(
        constant ds4_metal_args_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device       char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i00 = i0%args.ne00;
        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

// Host-visible F32 repeat used for HC expansion of embeddings.
template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;

// Qwen GQA expand: 4 kv heads (1024) -> 24 heads (6144), head_dim 256, factor 6.
struct ds4_gqa_args {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
};

kernel void kernel_gqa_expand_f32(
        device const float * src [[buffer(0)]],
        device float * dst [[buffer(1)]],
        constant ds4_gqa_args & args [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_head * args.head_dim;
    if (gid >= total) return;
    const uint h = gid / args.head_dim;
    const uint off = gid % args.head_dim;
    const uint group = args.n_head / args.n_head_kv;
    const uint kv_h = h / group;
    dst[gid] = src[kv_h * args.head_dim + off];
}

// Fill with zero (or arbitrary value) for attn_out zero path
kernel void kernel_fill_f32(
        device float * dst [[buffer(0)]],
        constant float & value [[buffer(1)]],
        constant uint & n [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = value;
}
