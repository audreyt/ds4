// Experimental M5 INT8 TensorOps probes.
//
// This raw INT8 GEMM kernel is a ds4-local probe inspired by Cider's public
// M5 TensorOps work.  It deliberately avoids model integration, activation
// quantization, and dequantization so tests can first prove the primitive.

#ifdef DS4_METAL_HAS_TENSOR

static constant short DS4_INT8_FRAG_ELEMS = 8;
static constant short DS4_INT8_FRAG_COLS = 4;
static constant short DS4_INT8_FRAG_ROW_JUMP = 8;

static inline short2 ds4_int8_frag_coord(ushort lane) {
    short qid = short(lane >> 2);
    short fm = ((qid & 4) | ((short(lane) >> 1) & 3));
    short fn = ((qid & 2) | (short(lane) & 1)) * 4;
    return short2(fn, fm);
}

static inline void ds4_int8_frag_load(
        thread int8_t       *dst,
        device const int8_t *src,
        int                  ld,
        short2               sc,
        short                off_m,
        short                off_n) {
    src += (sc.y + off_m) * ld + (sc.x + off_n);
    for (short i = 0; i < 2; i++) {
        for (short j = 0; j < DS4_INT8_FRAG_COLS; j++) {
            dst[i * DS4_INT8_FRAG_COLS + j] =
                src[(i * DS4_INT8_FRAG_ROW_JUMP) * ld + j];
        }
    }
}

static inline void ds4_int8_frag_store_i32(
        thread const int32_t *src,
        device int32_t      *dst,
        int                  ld,
        short2               sc,
        short                off_m,
        short                off_n,
        uint                 M,
        uint                 N,
        uint                 m_base,
        uint                 n_base) {
    for (short i = 0; i < 2; i++) {
        for (short j = 0; j < DS4_INT8_FRAG_COLS; j++) {
            uint mi = m_base + sc.y + off_m + i * DS4_INT8_FRAG_ROW_JUMP;
            uint ni = n_base + sc.x + off_n + j;
            if (mi < M && ni < N) {
                dst[(sc.y + off_m + i * DS4_INT8_FRAG_ROW_JUMP) * ld +
                    (sc.x + off_n + j)] = src[i * DS4_INT8_FRAG_COLS + j];
            }
        }
    }
}

kernel void kernel_int8_matmul_i32_mpp_probe(
        device const int8_t *A [[buffer(0)]],
        device const int8_t *B [[buffer(1)]],
        device int32_t      *C [[buffer(2)]],
        constant uint       &M [[buffer(3)]],
        constant uint       &N [[buffer(4)]],
        constant uint       &K [[buffer(5)]],
        uint2                tgid [[threadgroup_position_in_grid]],
        uint                 sgid [[simdgroup_index_in_threadgroup]],
        uint                 lane [[thread_index_in_simdgroup]]) {
    constexpr int BM = 32;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int WM = 1;
    constexpr int WN = 4;
    constexpr short TM = 2;
    constexpr short TN = 2;
    constexpr short TK = 2;

    const uint tile_m = tgid.y;
    const uint tile_n = tgid.x;
    const uint sg_row = sgid / WN;
    const uint sg_col = sgid % WN;
    const uint m_base = tile_m * BM + sg_row * (BM / WM);
    const uint n_base = tile_n * BN + sg_col * (BN / WN);
    if (m_base >= M || n_base >= N) return;

    short2 sc = ds4_int8_frag_coord(ushort(lane));

    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> mm;

    auto ct_a = mm.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ct_b = mm.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto ct_c = mm.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

    int32_t c_frag[TM * TN][DS4_INT8_FRAG_ELEMS];
    for (short f = 0; f < TM * TN; f++) {
        for (short i = 0; i < DS4_INT8_FRAG_ELEMS; i++) c_frag[f][i] = 0;
    }

    device const int8_t *tile_A = A + m_base * K;
    device const int8_t *tile_B = B + n_base * K;

    for (uint kk0 = 0; kk0 < K; kk0 += BK) {
        int8_t a_frag[TM][TK][DS4_INT8_FRAG_ELEMS];
        int8_t b_frag[TN][TK][DS4_INT8_FRAG_ELEMS];
        volatile int compiler_barrier;

        for (short mm_i = 0; mm_i < TM; mm_i++) {
            for (short kk_i = 0; kk_i < TK; kk_i++) {
                ds4_int8_frag_load(a_frag[mm_i][kk_i],
                                   tile_A + kk0,
                                   int(K),
                                   sc,
                                   short(mm_i * 16),
                                   short(kk_i * 16));
            }
        }
        for (short nn_i = 0; nn_i < TN; nn_i++) {
            for (short kk_i = 0; kk_i < TK; kk_i++) {
                ds4_int8_frag_load(b_frag[nn_i][kk_i],
                                   tile_B + kk0,
                                   int(K),
                                   sc,
                                   short(nn_i * 16),
                                   short(kk_i * 16));
            }
        }

        for (short mm_i = 0; mm_i < TM; mm_i++) {
            for (short nn_i = 0; nn_i < TN; nn_i += 2) {
                for (short kk_i = 0; kk_i < TK; kk_i++) {
                    for (short i = 0; i < DS4_INT8_FRAG_ELEMS; i++) ct_a[i] = a_frag[mm_i][kk_i][i];
                    for (short i = 0; i < DS4_INT8_FRAG_ELEMS; i++) {
                        ct_b[i] = b_frag[nn_i][kk_i][i];
                        ct_b[DS4_INT8_FRAG_ELEMS + i] = b_frag[nn_i + 1][kk_i][i];
                    }
                    const short c0 = mm_i * TN + nn_i;
                    const short c1 = c0 + 1;
                    for (short i = 0; i < DS4_INT8_FRAG_ELEMS; i++) {
                        ct_c[i] = c_frag[c0][i];
                        ct_c[DS4_INT8_FRAG_ELEMS + i] = c_frag[c1][i];
                    }
                    mm.run(ct_a, ct_b, ct_c);
                    for (short i = 0; i < DS4_INT8_FRAG_ELEMS; i++) {
                        c_frag[c0][i] = ct_c[i];
                        c_frag[c1][i] = ct_c[DS4_INT8_FRAG_ELEMS + i];
                    }
                }
            }
        }
        (void)compiler_barrier;
    }

    device int32_t *tile_C = C + m_base * N + n_base;
    for (short mm_i = 0; mm_i < TM; mm_i++) {
        for (short nn_i = 0; nn_i < TN; nn_i++) {
            ds4_int8_frag_store_i32(c_frag[mm_i * TN + nn_i],
                                    tile_C,
                                    int(N),
                                    sc,
                                    short(mm_i * 16),
                                    short(nn_i * 16),
                                    M,
                                    N,
                                    m_base,
                                    n_base);
        }
    }
}

#endif
