typedef long long i64;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef signed char i8;

#ifndef MARK_VALUE
#define MARK_VALUE (-2147483647 - 1)
#endif
#ifndef STOP
#define STOP 0
#endif
#ifndef STOP_PART
#define STOP_PART 0
#endif
#ifndef PART_ONLY
#define PART_ONLY (-1)
#endif
#define HALT(k) (STOP == (k) && p == STOP_PART)
#ifndef RANK_MAX
#define RANK_MAX 2048
#endif
#ifndef PROF
#define PROF 0
#endif
#ifndef CHECKED
#define CHECKED 0
#endif
#ifndef UN_WIDE
#define UN_WIDE 0
#endif
#ifndef POISON
#define POISON 0
#endif
#ifndef CORRUPT
#define CORRUPT 0
#endif
#ifndef NOUNROLL
#define NOUNROLL 0
#endif
#ifndef SYNCWARP
#define SYNCWARP 0
#endif
#ifndef FENCE
#define FENCE 0
#endif
#ifndef BARRIERS
#define BARRIERS 0
#endif
#ifndef LAYOUT_PROBE
#define LAYOUT_PROBE 0
#endif
#ifndef VERIFY
#define VERIFY 0
#endif
#ifndef SCAN_SMEM
#define SCAN_SMEM 0
#endif
#ifndef BARRIER_ASM
#define BARRIER_ASM 0
#endif
#ifndef SELFTEST
#define SELFTEST 0
#endif
#ifndef SELFTEST_SITES
#define SELFTEST_SITES 1
#endif
#ifndef SELFTEST_PARTS
#define SELFTEST_PARTS 7
#endif
#ifndef SELFTEST_MASKS
#define SELFTEST_MASKS 0
#endif
#define STPARTS(site) (SELFTEST_MASKS ? ((SELFTEST_MASKS >> (3 * (site))) & 7) :                        (((SELFTEST_SITES >> (site)) & 1) ? SELFTEST_PARTS : 0))
#ifndef SYNC_WARP
#define SYNC_WARP 0
#endif
#if BARRIER_ASM
#define BAR_RAW() asm volatile("bar.sync 0, %0;" :: "r"(THREADS) : "memory")
#else
#define BAR_RAW() __syncthreads()
#endif
#if SYNC_WARP
#define BAR() do { __syncwarp(); BAR_RAW(); } while (0)
#else
#define BAR() BAR_RAW()
#endif
#ifndef LRU_STAGED
#define LRU_STAGED 0
#endif
#if FENCE
#define SYNC() do { __threadfence(); BAR(); } while (0)
#else
#define SYNC() BAR()
#endif
#if BARRIERS
#define EXTRA() SYNC()
#else
#define EXTRA() do {} while (0)
#endif
#define STEST(site) \
    do { \
    if (SELFTEST && STPARTS(site)) { \
    int wrong_ = 0; \
    for (int it_ = 0; it_ < SELFTEST; it_++) { \
    if ((STPARTS(site) & 1) && (tid & 31) == 0) stest[tid >> 5] = it_ * 64 + (tid >> 5) + (site) * 7; \
    if (STPARTS(site) & 2) \
    __nanosleep((((unsigned)(tid >> 5) * 2654435761u) ^ ((unsigned)it_ * 40503u)) % 500u); \
    if (STPARTS(site) & 4) SYNC(); \
    if (STPARTS(site) & 1) { \
    const int peer_ = ((tid >> 5) + 1 + it_ % 31) & 31; \
    if (stest[peer_] != it_ * 64 + peer_ + (site) * 7) wrong_++; \
    } \
    if (STPARTS(site) & 4) SYNC(); \
    } \
    if (wrong_ && (STPARTS(site) & 5) == 5 && atomicAdd(&red[NW], 1) < 3) \
    printf("RESOLVE_CUDA_VERIFY selftest site=%d block=%d tid=%d wrong=%d\n", (site), lane, tid, wrong_); \
    } \
    } while (0)
#ifndef CLEAR_ATOMIC
#define CLEAR_ATOMIC 0
#endif
#ifndef TABLE_ATOMIC
#define TABLE_ATOMIC 0
#endif
#ifndef TABLE_SORT
#define TABLE_SORT 0
#endif
#ifndef BOUNDED
#define BOUNDED 1
#endif
#ifndef WARP_LOOPS
#define WARP_LOOPS 1
#endif
#ifndef SORT_REGS
#define SORT_REGS 1
#endif
#if NOUNROLL
#define UNR _Pragma("unroll 1")
#else
#define UNR _Pragma("unroll")
#endif
#define TICK(k) do { if (PROF && tid == 0) { long long now_ = clock64(); ((long long *)a.prof)[lane * 16 + (k)] += now_ - last_; last_ = now_; } } while (0)
#define NW (THREADS / 32)
#define PW ((KB + THREADS - 1) / THREADS)
#define RPT ((HB + THREADS - 1) / THREADS)
#define HS (2 * KB)
#define SORT_PAD 0x7fffffffffffffffLL

#define S_REQ 1
#define S_STATE 2
#define S_UNION 3
#define S_SRC 4
#define S_RES 5
#define S_HOT_HIT 6
#define S_HOT_EVICT 7
#define S_LRU_R 8
#define S_LRU_W 9
#define S_DGI_R 10
#define S_DGI_W 11
#define S_MARKS_W 12
#define S_MARKS_R 13
#define S_DEVS_W 14
#define S_HOT_OUT 15
#define S_ATTN 16
#define S_FRESH 17
#define S_HOST_OUT 18
#define S_DEV_OUT 19
#define S_COUNTS 20
#define S_PLAN_HOST 21
#define S_PLAN_DEV 22
#define S_PLAN_COUNT 23
#define S_IDLE 24
#define S_LAYOUT 25
#define S_WIDTH 26
#define S_GS 27
#define S_TAB 28
#define S_EV 29
#define S_SRT 30
#define S_PIECE 31
#define S_TABCLR 32
#define S_TABINS 33

//ARGS//

#if CHECKED
__device__ __noinline__ void note_violation(const Args &a, int site, i64 idx, i64 ext) {
    if (atomicAdd((u64 *)a.viol, 1ull) < 32ull)
        printf("RESOLVE_CUDA_OOB site=%d index=%lld extent=%lld block=%d thread=%d\n", site, idx, ext,
               (int)blockIdx.x, (int)threadIdx.x);
}
#else
__device__ __forceinline__ void note_violation(const Args &a, int site, i64 idx, i64 ext) {
    atomicAdd((u64 *)a.viol, 1ull);
}
#endif
__device__ __forceinline__ bool in_range(const Args &a, int site, i64 idx, i64 ext) {
    if (__builtin_expect((u64)idx < (u64)ext, 1)) return true;
    note_violation(a, site, idx, ext);
    return false;
}
#if CHECKED
#define IN(site, idx, ext) in_range(a, site, (i64)(idx), (i64)(ext))
#else
#define IN(site, idx, ext) true
#endif
#define SAFE(site, idx, ext) in_range(a, site, (i64)(idx), (i64)(ext))
#define LD(site, p, idx, ext, other) (IN(site, (idx), (ext)) ? (p)[(idx)] : (other))
#define ST(site, p, idx, ext, v) do { if (IN(site, (idx), (ext))) (p)[(idx)] = (v); } while (0)
#if BOUNDED
#define BIN(site, idx, ext) SAFE(site, idx, ext)
#else
#define BIN(site, idx, ext) IN(site, idx, ext)
#endif
#define BLD(site, p, idx, ext, other) (BIN(site, (idx), (ext)) ? (p)[(idx)] : (other))
#define BST(site, p, idx, ext, v) do { if (BIN(site, (idx), (ext))) (p)[(idx)] = (v); } while (0)

#if UN_WIDE
typedef u64 tok_t;
#else
typedef unsigned tok_t;
#endif

__device__ __forceinline__ int warp_scan(int v) {
#if SYNCWARP
    __syncwarp();
#endif
    const int l = threadIdx.x & 31;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, v, o);
        if (l >= o) v += n;
    }
    return v;
}

extern __shared__ __align__(16) unsigned char smem_raw[];

#if SCAN_SMEM
__device__ __forceinline__ int block_scan(int v, int *total, int *red) {
    int *sc = (int *)smem_raw + 2 * KB - THREADS;
    SYNC();
    sc[threadIdx.x] = v;
    SYNC();
    int off = 0, tot = 0;
    for (int k = 0; k < THREADS; k++) {
        const int x = sc[k];
        if (k < (int)threadIdx.x) off += x;
        tot += x;
    }
    SYNC();
    *total = tot;
    return off;
}
#else
__device__ __forceinline__ int block_scan(int v, int *total, int *red) {
    const int w = threadIdx.x >> 5;
    int inc = warp_scan(v);
    if ((threadIdx.x & 31) == 31) red[w] = inc;
    SYNC();
    int off = 0, tot = 0;
#pragma unroll
    for (int k = 0; k < NW; k++) {
        int x = red[k];
        if (k < w) off += x;
        tot += x;
    }
    SYNC();
    *total = tot;
    return off + inc - v;
}
#endif

__device__ __forceinline__ unsigned home_of(i64 g) {
    return ((unsigned)(g * 2654435761LL)) >> (32 - (LOG + 1));
}

__device__ __forceinline__ void pick_info(const Args &a, i64 req, int lane, int first, int i, i64 &g, int &resident,
                                          bool &pending, bool &done) {
    const i64 pidx = (i64)lane * a.union_stride + first + i;
    const i64 pick = (i64)LD(S_UNION, (const UN_T *)a.union_p, pidx, a.ext_union, (UN_T)-1);
    const tok_t token = pick < 0 ? (tok_t)0 : (tok_t)pick;
    const tok_t blk = token / SRC_BS;
    const bool found = pick >= 0 && req >= 0 && req < a.src_reqs && blk < (tok_t)a.src_blocks;
    const i64 phys = found ? (i64)LD(S_SRC, (const SR_T *)a.src_table, req * a.src_stride + (i64)blk, a.ext_src,
                                     (SR_T)0) : 0;
    g = (found && phys > 0) ? phys * SRC_BS + (i64)(token % SRC_BS) : -1;
    i64 res = -1;
#if HAS_RES
    {
        const tok_t rb = token / RES_BS;
        const bool rf = pick >= 0 && req >= 0 && req < a.res_reqs && rb < (tok_t)a.res_blocks;
        const i64 rp = rf ? (i64)LD(S_RES, (const RS_T *)a.res_table, req * a.res_stride + (i64)rb, a.ext_res,
                                    (RS_T)-1) : -1;
        res = (rf && rp != a.res_null && rp >= 0) ? rp * RES_BS + (i64)(token % RES_BS) : -1;
    }
#endif
    if (g >= a.host_total) g = -1;
    done = (res >= 0) || (g < 0);
    pending = !done;
    resident = res >= 0 ? (int)res : -1;
}

__device__ __forceinline__ bool lane_fits(const Args &a, int lane) {
    return (i64)lane < a.ext_req && (i64)lane < a.ext_pcount && (i64)(lane + 1) * KB <= a.ext_marks
#if HAS_IDLE
           && (i64)lane < a.ext_idle
#endif
        ;
}

__device__ __forceinline__ int attn_of(const Args &a, int hot) {
    return hot < 0 ? -1 : (int)((i64)((unsigned)hot / HOT_BS) * a.attn_block_stride + (unsigned)hot % HOT_BS);
}

__device__ __forceinline__ int hot_index(const Args &a, i64 req, int slot, int site) {
    const i64 at = req * a.hot_table_stride + (i64)((unsigned)slot / HOT_BS);
    i64 hb = -1;
    if (SAFE(site, at, a.ext_hot)) hb = (i64)((const HT_T *)a.hot_table)[at];
    return (int)(hb * HOT_BS + (i64)((unsigned)slot % HOT_BS));
}

extern "C" __global__ void __launch_bounds__(THREADS) one_resolve(const Args a) {
    __shared__ int red[NW + 2];
    __shared__ int stest[32];
    unsigned *tab = (unsigned *)smem_raw;
    int *ev = (int *)smem_raw;
    int *gs = (int *)(smem_raw + 4 * HS);
    i64 *srt = (i64 *)smem_raw;

    const int lane = blockIdx.x;
    const int tid = threadIdx.x;
#if VERIFY || SELFTEST
    if (tid == 0) {
        red[NW] = 0;
        red[NW + 1] = 0;
    }
    __syncthreads();
#endif
#if POISON
    for (int k = tid; k < 3 * KB; k += THREADS) ((unsigned *)smem_raw)[k] = 0x9e3779b9u * (unsigned)(k + 1) + (unsigned)lane;
    __syncthreads();
#endif
#if LAYOUT_PROBE
    if (lane == 0 && tid == 0) {
        unsigned dyn, total;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
        asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(total));
        printf("RESOLVE_CUDA_LAYOUT red=%u smem_raw=%u dynamic=%u total=%u KB=%d THREADS=%d HS=%d\n",
               (unsigned)__cvta_generic_to_shared(red), (unsigned)__cvta_generic_to_shared(smem_raw), dyn, total, KB,
               THREADS, HS);
    }
#endif
    if (!lane_fits(a, lane)) {
        if (tid == 0) note_violation(a, S_LAYOUT, lane, a.ext_req);
        return;
    }
    const i64 req = (i64)((const RQ_T *)a.request_ids)[lane];
    const bool known = req >= 0 && req < a.state_count;
    i64 state = -1;
    if (known) state = (i64)((const ST_T *)a.states)[req];
    if (state >= a.state_rows) {
        if (tid == 0) SAFE(S_STATE, state, a.state_rows);
        state = -1;
    }
    const i64 so = state * a.region;
    short *lru = (short *)a.lru;
    int *dgi = (int *)a.dgi;
    int *marks = (int *)a.marks;
    int *devs = (int *)a.devs;
    const i64 mb = (i64)lane * KB;
    i64 filled = 0;
    long long last_ = PROF ? clock64() : 0;

    for (int p = 0; p < a.parts; p++) {
        SYNC();
        if (PART_ONLY >= 0 && p != PART_ONLY) continue;
        STEST(0);
        const i64 *layout = (const i64 *)a.layout;
        const int first = (int)layout[p];
        const int width = (int)layout[p + 1] - first;
        const int wd = width < KB ? width : KB;
        if (tid == 0 && width > KB) SAFE(S_WIDTH, width, KB + 1);
        const i64 listed = layout[a.parts + 1 + p] * a.lanes + (i64)lane * width;
        const int count_at = p * a.lanes;
        int *hot_out = (int *)a.hot_out;
        const i64 arow = (i64)lane * a.attn_stride + first;
        int *attn = (int *)a.attn;
        const i64 frow = (i64)lane * a.fresh_stride + first;
        i8 *fresh = (i8 *)a.fresh;
        const i64 urow = (i64)lane * a.union_stride + first;
        const bool geom = first >= 0 && wd >= 0 && urow >= 0 && urow + wd <= a.ext_union && listed >= 0 &&
                          listed + wd <= a.ext_lists && arow >= 0 && arow + wd <= a.ext_attn &&
                          (!FRESH || (frow >= 0 && frow + wd <= a.ext_fresh)) &&
                          (i64)count_at + lane < a.ext_counts;
        if (!geom) {
            if (tid == 0) note_violation(a, S_LAYOUT, listed, a.ext_lists);
            continue;
        }

        bool blank = state < 0;
        if (!blank && SKIP) {
            int nn = 0;
            for (int i = tid; i < wd; i += THREADS)
                nn += ((i64)LD(S_UNION, (const UN_T *)a.union_p, (i64)lane * a.union_stride + first + i, a.ext_union,
                               (UN_T)-1) >= 0);
            int tot;
            block_scan(nn, &tot, red);
            blank = tot == 0;
        }
        if (blank) {
            for (int i = tid; i < wd; i += THREADS) {
                ST(S_HOT_OUT, hot_out, listed + i, a.ext_lists, -1);
                ST(S_ATTN, attn, arow + i, a.ext_attn, -1);
                if (FRESH) ST(S_FRESH, fresh, frow + i, a.ext_fresh, (i8)0);
            }
            if (tid == 0) ST(S_COUNTS, (int *)a.counts_out, count_at + lane, a.ext_counts, 0);
            continue;
        }

        u64 pend = 0;
        int dn = 0;
        #pragma unroll
        for (int j = 0; j < PW; j++) {
            const int i = tid * PW + j;
            if (i < wd) {
                i64 g;
                int res;
                bool pd, dd;
                pick_info(a, req, lane, first, i, g, res, pd, dd);
                gs[i] = pd ? (int)g : -1;
                if (pd) pend |= 1ull << j;
                dn += dd;
            }
        }
        int settled;
        block_scan(dn, &settled, red);
        TICK(0);
        if (HALT(1)) return;

        if (SKIP && settled == width) {
            for (int i = tid; i < wd; i += THREADS) {
                i64 g;
                int res;
                bool pd, dd;
                pick_info(a, req, lane, first, i, g, res, pd, dd);
                ST(S_HOT_OUT, hot_out, listed + i, a.ext_lists, res);
                ST(S_ATTN, attn, arow + i, a.ext_attn, attn_of(a, res));
                if (FRESH) ST(S_FRESH, fresh, frow + i, a.ext_fresh, (i8)0);
            }
            if (tid == 0) {
                ST(S_COUNTS, (int *)a.counts_out, count_at + lane, a.ext_counts, 0);
                atomicAdd((u64 *)a.stats, (u64)settled);
            }
            continue;
        }

#if TABLE_SORT
#if SORT_REGS && KB >= THREADS
        const int np = KB;
        i64 v[PW];
#pragma unroll
        for (int e = 0; e < PW; e++) {
            const int t = tid * PW + e;
            const int g = t < wd ? gs[t] : -1;
            v[e] = g >= 0 ? (((i64)g << LOG) | (i64)t) : SORT_PAD;
        }
        for (int i = tid; i < wd; i += THREADS) ST(S_MARKS_W, marks, mb + i, a.ext_marks, MARK_VALUE);
        for (int k = 2; k <= KB; k <<= 1) {
            if ((k >> 1) >= 32 * PW) {
#pragma unroll
                for (int e = 0; e < PW; e++) srt[tid * PW + e] = v[e];
                SYNC();
                for (int j = k >> 1; j >= 32 * PW; j >>= 1) {
                    for (int q = tid; q < (KB >> 1); q += THREADS) {
                        const int t = ((q & ~(j - 1)) << 1) | (q & (j - 1));
                        const int x = t | j;
                        const i64 u = srt[t], w = srt[x];
                        if ((u > w) == ((t & k) == 0)) {
                            srt[t] = w;
                            srt[x] = u;
                        }
                    }
                    SYNC();
                }
#pragma unroll
                for (int e = 0; e < PW; e++) v[e] = srt[tid * PW + e];
            }
            for (int j = ((k >> 1) < 16 * PW ? (k >> 1) : 16 * PW); j >= PW; j >>= 1) {
                const int across = j / PW;
#pragma unroll
                for (int e = 0; e < PW; e++) {
                    const int t = tid * PW + e;
                    const i64 p = __shfl_xor_sync(0xffffffffu, v[e], across);
                    const bool low = ((t & j) == 0) == ((t & k) == 0);
                    v[e] = low ? (p < v[e] ? p : v[e]) : (p > v[e] ? p : v[e]);
                }
            }
#pragma unroll
            for (int j = PW >> 1; j > 0; j >>= 1) {
                if (j < k) {
#pragma unroll
                    for (int e = 0; e < PW; e++) {
                        if ((e & j) == 0) {
                            const int f = e | j;
                            const bool up = ((tid * PW + e) & k) == 0;
                            if ((v[e] > v[f]) == up) {
                                const i64 u = v[e];
                                v[e] = v[f];
                                v[f] = u;
                            }
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int e = 0; e < PW; e++) srt[tid * PW + e] = v[e];
        SYNC();
#else
        int np = 1;
        while (np < wd) np <<= 1;
        for (int t = tid; t < np; t += THREADS) {
            const int g = t < wd ? gs[t] : -1;
            srt[t] = g >= 0 ? (((i64)g << LOG) | (i64)t) : SORT_PAD;
        }
        for (int i = tid; i < wd; i += THREADS) ST(S_MARKS_W, marks, mb + i, a.ext_marks, MARK_VALUE);
        SYNC();
        for (int k = 2; k <= np; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int q = tid; q < (np >> 1); q += THREADS) {
                    const int t = ((q & ~(j - 1)) << 1) | (q & (j - 1));
                    const int x = t | j;
                    const i64 u = srt[t], v = srt[x];
                    if ((u > v) == ((t & k) == 0)) {
                        srt[t] = v;
                        srt[x] = u;
                    }
                }
                SYNC();
            }
        }
#endif
        TICK(1);
        if (HALT(2)) return;

        int sv[RPT];
        int cd[RPT];
        u64 hm = 0;
        int hc = 0;
#pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int pos = tid * RPT + r;
            sv[r] = pos < a.region ? (int)LD(S_LRU_R, lru, so + pos, a.ext_lru, (short)-1) : -1;
        }
#pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int slot = sv[r];
            const bool sane = (tid * RPT + r) < a.region && slot >= 0 && slot < a.region;
            cd[r] = sane ? LD(S_DGI_R, dgi, so + slot, a.ext_dgi, -1) : -1;
        }
#pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int cached = cd[r];
            const i64 target = (i64)(cached < 0 ? 0 : cached) << LOG;
            int at = 0;
            for (int step = np >> 1; step > 0; step >>= 1)
                if (srt[at + step - 1] < target) at += step;
            const i64 near = srt[at];
            if (cached >= 0 && (near >> LOG) == (i64)cached) {
                const int col = (int)(near & (KB - 1));
                if (BIN(S_GS, col, wd)) {
                    ST(S_MARKS_W, marks, mb + col, a.ext_marks, hot_index(a, req, sv[r], S_HOT_HIT));
                    hm |= 1ull << r;
                    hc++;
                }
            }
        }
#else
        for (int k = tid; k < HS; k += THREADS) {
#if CLEAR_ATOMIC || TABLE_ATOMIC
            atomicExch(&tab[k], 0u);
#else
            tab[k] = 0u;
#endif
        }
        for (int i = tid; i < wd; i += THREADS) ST(S_MARKS_W, marks, mb + i, a.ext_marks, MARK_VALUE);
        SYNC();
        EXTRA();
#if CHECKED
        for (int k = tid; k < HS; k += THREADS)
            if (tab[k] != 0u) note_violation(a, S_TABCLR, k, (i64)tab[k]);
        SYNC();
#endif

#if WARP_LOOPS
        UNR
        for (int j = 0; j < PW; j++) {
            bool live = (pend >> j) & 1;
            const int i = tid * PW + j;
            const int g = live ? gs[i] : 0;
            const unsigned e = (unsigned)(i + 1);
            unsigned h = home_of(g);
            int tries = 0;
            while (__any_sync(0xffffffffu, live)) {
                if (live) {
                    const unsigned old = atomicCAS(&tab[h], 0u, e);
                    if (old == 0 || !SAFE(S_TAB, (i64)old - 1, wd) || ++tries > HS) {
                        live = false;
                    } else if (gs[old - 1] == g) {
                        atomicMin(&tab[h], e);
                        live = false;
                    } else {
                        h = (h + 1) & (HS - 1);
                    }
                }
            }
        }
#else
        UNR
        for (int j = 0; j < PW; j++) {
            if (!((pend >> j) & 1)) continue;
            const int i = tid * PW + j;
            const int g = gs[i];
            const unsigned e = (unsigned)(i + 1);
            unsigned h = home_of(g);
            for (;;) {
                const unsigned old = atomicCAS(&tab[h], 0u, e);
                if (old == 0) break;
                if (!SAFE(S_TAB, (i64)old - 1, wd)) break;
                if (gs[old - 1] == g) {
                    atomicMin(&tab[h], e);
                    break;
                }
                h = (h + 1) & (HS - 1);
            }
        }
#endif
        SYNC();
        EXTRA();
        STEST(1);
#if CHECKED
        for (int k = tid; k < HS; k += THREADS)
            if (tab[k] > (unsigned)wd) note_violation(a, S_TABINS, k, (i64)tab[k]);
        SYNC();
#endif
#if CORRUPT
        if (tid < CORRUPT) tab[(tid * 2654435761u) & (HS - 1)] = 0x9e3779b9u + (unsigned)tid;
        SYNC();
#endif
        TICK(1);
        if (HALT(2)) return;
        EXTRA();

        int sv[RPT];
        int cd[RPT];
        u64 hm = 0;
        int hc = 0;
#pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int pos = tid * RPT + r;
            sv[r] = pos < a.region ? (int)LD(S_LRU_R, lru, so + pos, a.ext_lru, (short)-1) : -1;
        }
#pragma unroll
        for (int r = 0; r < RPT; r++) {
            const int slot = sv[r];
            const bool sane = (tid * RPT + r) < a.region && slot >= 0 && slot < a.region;
            cd[r] = sane ? LD(S_DGI_R, dgi, so + slot, a.ext_dgi, -1) : -1;
        }
#if WARP_LOOPS
        UNR
        for (int r = 0; r < RPT; r++) {
            const int cached = cd[r];
            bool live = cached >= 0;
            unsigned h = home_of(live ? cached : 0);
            int tries = 0;
            while (__any_sync(0xffffffffu, live)) {
                if (live) {
                    const unsigned e = tab[h];
                    if (e == 0 || !SAFE(S_GS, (i64)e - 1, wd) || ++tries > HS) {
                        live = false;
                    } else if (gs[e - 1] == cached) {
                        ST(S_MARKS_W, marks, mb + (e - 1), a.ext_marks, hot_index(a, req, sv[r], S_HOT_HIT));
                        hm |= 1ull << r;
                        hc++;
                        live = false;
                    } else {
                        h = (h + 1) & (HS - 1);
                    }
                }
            }
        }
#else
        UNR
        for (int r = 0; r < RPT; r++) {
            const int cached = cd[r];
            if (cached >= 0) {
                unsigned h = home_of(cached);
                for (;;) {
#if TABLE_ATOMIC
                    const unsigned e = atomicOr(&tab[h], 0u);
#else
                    const unsigned e = tab[h];
#endif
                    if (e == 0) break;
                    if (!SAFE(S_GS, (i64)e - 1, wd)) break;
                    if (gs[e - 1] == cached) {
                        ST(S_MARKS_W, marks, mb + (e - 1), a.ext_marks, hot_index(a, req, sv[r], S_HOT_HIT));
                        hm |= 1ull << r;
                        hc++;
                        break;
                    }
                    h = (h + 1) & (HS - 1);
                }
            }
        }
#endif
#endif
        EXTRA();
        STEST(2);
        int hits;
        const int hbase = block_scan(hc, &hits, red);
#if VERIFY
        {
            int *hs = (int *)smem_raw + 2 * KB - THREADS;
            SYNC();
            hs[tid] = hc;
            SYNC();
            int ex = 0;
            for (int k = 0; k < tid; k++) ex += hs[k];
            if ((ex != hbase || (int)__popcll(hm) != hc) && atomicAdd(&red[NW], 1) < 3)
                printf("RESOLVE_CUDA_VERIFY scan block=%d tid=%d hbase=%d expected=%d hc=%d popcount=%d hits=%d\n",
                       lane, tid, hbase, ex, hc, (int)__popcll(hm), hits);
            SYNC();
        }
#endif
        const int misses_stat = width - hits - settled;
        const int olds = a.region - hits;
        const int rotate = misses_stat < 0 ? 0 : (misses_stat > olds ? olds : misses_stat);

        TICK(2);
        if (HALT(3)) return;
        int mk[PW];
        u64 miss = 0;
        int mc = 0;
        #pragma unroll
        for (int j = 0; j < PW; j++) {
            const int i = tid * PW + j;
            if (i < wd && ((pend >> j) & 1)) {
                const int mark = LD(S_MARKS_R, marks, mb + i, a.ext_marks, MARK_VALUE);
                mk[j] = mark;
                if (mark == MARK_VALUE) {
                    miss |= 1ull << j;
                    mc++;
                }
            }
        }
#if VERIFY
        for (int r = 0; r < RPT; r++) {
            const int pos = tid * RPT + r;
            if (pos >= a.region) break;
            const int now = (int)LD(S_LRU_R, lru, so + pos, a.ext_lru, (short)-1);
            if (now != sv[r] && atomicAdd(&red[NW + 1], 1) < 4)
                printf("RESOLVE_CUDA_VERIFY row_changed block=%d tid=%d pos=%d was=%d now=%d\n", lane, tid, pos, sv[r], now);
        }
#endif
        EXTRA();
        STEST(3);
        int mpick;
        const int mbase = block_scan(mc, &mpick, red);
        TICK(3);
        EXTRA();
        STEST(4);

#if LRU_STAGED
        short *stg = (short *)smem_raw + 2 * KB;
        const bool staged = (a.region & 1) == 0 && (a.region >> 1) <= KB - 2 * THREADS && so >= 0 &&
                            so + a.region <= a.ext_lru && (((u64)a.lru) & 3) == 0;
#endif
        {
            const int pos0 = tid * RPT;
            int hr = hbase;
            int kr = pos0 - hbase;
#pragma unroll
            for (int r = 0; r < RPT; r++) {
                if (pos0 + r >= a.region) break;
                const int slot = sv[r];
                int npos;
                if ((hm >> r) & 1) {
                    npos = olds + hr;
                    const int back = a.region - 1 - hr;
                    if (back < mpick) BST(S_EV, ev, back, 2 * KB, slot);
                    hr++;
                } else {
                    const int k = kr;
                    kr++;
                    if (k < mpick) BST(S_EV, ev, k, 2 * KB, slot);
                    npos = k >= rotate ? k - rotate : (olds - rotate) + k;
                }
                if (npos >= 0 && npos < a.region) {
#if LRU_STAGED
                    if (staged) stg[npos] = (short)slot;
                    else
#endif
                    ST(S_LRU_W, lru, so + npos, a.ext_lru, (short)slot);
                }
            }
        }
#if LRU_STAGED
        if (staged) {
            SYNC();
            int *dst = (int *)(lru + so);
            const int *src = (const int *)stg;
            for (int w = tid; w < (a.region >> 1); w += THREADS) dst[w] = src[w];
        }
#endif
        SYNC();
        TICK(4);
        STEST(5);
#if VERIFY
        {
            int *vf = (int *)smem_raw + 2 * KB - 2 * THREADS;
            const int words = (a.region + 31) >> 5;
            for (int w = tid; w < words + 8; w += THREADS) vf[w] = 0;
            SYNC();
            if (tid == 0) {
                vf[2] = hits;
                vf[words + 4] = settled;
                vf[words + 5] = rotate;
            }
            SYNC();
            int dup = 0, bad = 0;
            for (int r = 0; r < RPT; r++) {
                const int pos = tid * RPT + r;
                if (pos >= a.region) break;
                const int slot = (int)LD(S_LRU_R, lru, so + pos, a.ext_lru, (short)-1);
                if (slot < 0 || slot >= a.region) {
                    bad++;
                    continue;
                }
                const int old = atomicOr(&vf[4 + (slot >> 5)], 1 << (slot & 31));
                if ((old >> (slot & 31)) & 1) {
                    dup++;
                    if (atomicAdd(&vf[0], 1) < 3)
                        printf("RESOLVE_CUDA_VERIFY duplicate block=%d tid=%d pos=%d slot=%d hbase=%d hits=%d olds=%d rotate=%d\n",
                               lane, tid, pos, slot, hbase, hits, olds, rotate);
                }
            }
            if (bad && atomicAdd(&vf[1], 1) < 3)
                printf("RESOLVE_CUDA_VERIFY out_of_range block=%d tid=%d bad=%d\n", lane, tid, bad);
            if ((settled != vf[words + 4] || rotate != vf[words + 5]) && atomicAdd(&vf[words + 6], 1) < 3)
                printf("RESOLVE_CUDA_VERIFY settled_or_rotate_differ block=%d tid=%d settled=%d first=%d rotate=%d first=%d\n",
                       lane, tid, settled, vf[words + 4], rotate, vf[words + 5]);
            if (hits != vf[2] && atomicAdd(&vf[3], 1) < 3)
                printf("RESOLVE_CUDA_VERIFY hits_differ block=%d tid=%d hits=%d first=%d\n", lane, tid, hits, vf[2]);
            SYNC();
            if (tid == 0 && (vf[0] || vf[1] || vf[3]))
                printf("RESOLVE_CUDA_VERIFY block=%d duplicates=%d range=%d hits_differ=%d hits=%d olds=%d rotate=%d mpick=%d settled=%d width=%d\n",
                       lane, vf[0], vf[1], vf[3], hits, olds, rotate, mpick, settled, width);
        }
#endif
        if (HALT(4)) return;

        int count = misses_stat;
#if HAS_IDLE
        if (LD(S_IDLE, (const unsigned char *)a.idle, lane, a.ext_idle, (unsigned char)0) != 0) count = 0;
#endif
        u64 good = 0;
        int dv[PW];
        int mrun = mbase;
        int mi[PW];
        #pragma unroll
        for (int j = 0; j < PW; j++) {
            const int i = tid * PW + j;
            dv[j] = -1;
            mi[j] = 0;
            int hot_i = -1;
            if (i < wd) {
                const bool pd = (pend >> j) & 1;
                const bool ms = (miss >> j) & 1;
                int res = -1;
#if HAS_RES
                {
                    i64 g;
                    bool p2, d2;
                    pick_info(a, req, lane, first, i, g, res, p2, d2);
                }
#endif
                if (pd && !ms) {
                    hot_i = mk[j];
                } else if (ms) {
                    const int m = mrun++;
                    mi[j] = m;
                    const int evict = BLD(S_EV, ev, m, 2 * KB, -1);
                    const bool gd = evict >= 0 && evict < a.region;
                    int landed = 0;
                    if (gd) {
                        landed = hot_index(a, req, evict, S_HOT_EVICT);
                        good |= 1ull << j;
                        dv[j] = landed;
                        ST(S_DGI_W, dgi, so + evict, a.ext_dgi, gs[i]);
                    }
                    hot_i = gd ? landed : res;
                } else {
                    hot_i = res;
                }
                ST(S_HOT_OUT, hot_out, listed + i, a.ext_lists, hot_i);
                ST(S_ATTN, attn, arow + i, a.ext_attn, attn_of(a, hot_i));
                if (FRESH) ST(S_FRESH, fresh, frow + i, a.ext_fresh, (i8)((ms && ((good >> j) & 1)) ? 1 : 0));
                if (ms && BIN(S_HOST_OUT, mi[j], wd)) {
                    ST(S_HOST_OUT, (int *)a.host_out, listed + mi[j], a.ext_lists, gs[i]);
                    ST(S_DEV_OUT, (int *)a.dev_out, listed + mi[j], a.ext_lists, dv[j]);
                }
            }
        }
        if (tid == 0) {
            atomicAdd((u64 *)a.stats, (u64)(i64)hits);
            atomicAdd((u64 *)a.stats + 1, (u64)(i64)misses_stat);
            ST(S_COUNTS, (int *)a.counts_out, count_at + lane, a.ext_counts, count);
#if TALLY
            atomicAdd((u64 *)a.tally, (u64)(i64)count);
            atomicAdd((u64 *)a.tally + 1, (u64)(count > 0 ? 1 : 0));
#endif
        }

        u64 take = 0;
#pragma unroll
        for (int j = 0; j < PW; j++)
            if (((good >> j) & 1) && count > 0) take |= 1ull << j;
#if GUARD
        {
            for (int i = tid; i < KB; i += THREADS) ST(S_MARKS_W, marks, mb + i, a.ext_marks, 0);
            SYNC();
            #pragma unroll
        for (int j = 0; j < PW; j++) {
                const int i = tid * PW + j;
                if (i < wd && ((miss >> j) & 1) && mi[j] < count && mi[j] < wd) {
                    const int g = gs[i];
                    const int dev = dv[j];
                    if (g >= 0 && (g >= a.host_total || dev < 0 || dev >= a.hot_total))
                        ST(S_MARKS_W, marks, mb + mi[j] / a.piece, a.ext_marks, 1);
                }
            }
            SYNC();
            int flagged_sum = 0;
            for (int i = tid; i < KB; i += THREADS)
                if ((i64)i * a.piece < count) flagged_sum += LD(S_MARKS_R, marks, mb + i, a.ext_marks, 0);
            #pragma unroll
        for (int j = 0; j < PW; j++) {
                const int i = tid * PW + j;
                if (i < wd && ((miss >> j) & 1) && mi[j] < wd &&
                    LD(S_MARKS_R, marks, mb + mi[j] / a.piece, a.ext_marks, 0) != 0)
                    take &= ~(1ull << j);
            }
            int gt;
            block_scan(flagged_sum, &gt, red);
            if (tid == 0) atomicAdd((u64 *)a.guard, (u64)(i64)gt);
        }
#endif
        TICK(5);
        if (HALT(5)) return;
        int cn = 0;
        i64 ck[PW];
        #pragma unroll
        for (int j = 0; j < PW; j++) {
            const int i = tid * PW + j;
            if (i < wd) {
                const bool tk = (take >> j) & 1;
                ST(S_DEVS_W, devs, mb + i, a.ext_marks, tk ? dv[j] : -1);
                if (tk && dv[j] >= 0) ck[cn++] = ((i64)gs[i] << LOG) | (i64)i;
            }
        }
        int total;
        const int cbase = block_scan(cn, &total, red);
        SYNC();
        TICK(6);
        if (HALT(6)) return;
        const i64 prow = (i64)lane * a.plan_stride + filled;
        const bool plan_ok = (i64)total <= KB && (i64)total <= a.plan_stride - filled && prow + total <= a.ext_plan;
        if (total > 0 && !plan_ok && tid == 0) note_violation(a, S_PLAN_HOST, prow + total, a.ext_plan);
        if (total > 0 && plan_ok) {
            int np = 1;
            while (np < total) np <<= 1;
#pragma unroll
            for (int c = 0; c < PW; c++)
                if (c < cn) BST(S_SRT, srt, cbase + c, total, ck[c]);
            SYNC();
            if (total <= RANK_MAX) {
                for (int e0 = 0; e0 < total; e0 += THREADS) {
                    const int e = e0 + tid;
                    const bool on = e < total;
                    if (__any_sync(0xffffffffu, on)) {
                        const i64 mine = srt[on ? e : 0];
                        int rank = 0;
#pragma unroll 16
                        for (int t = 0; t < total; t++) rank += srt[t] < mine;
                        if (on) {
                            ST(S_PLAN_HOST, (int *)a.plan_host, prow + rank, a.ext_plan, (int)(mine >> LOG));
                            ST(S_PLAN_DEV, (int *)a.plan_dev, prow + rank, a.ext_plan,
                               LD(S_DEVS_W, devs, mb + (int)(mine & (KB - 1)), a.ext_marks, -1));
                        }
                    }
                }
            } else {
                for (int t = total + tid; t < np; t += THREADS) srt[t] = SORT_PAD;
                SYNC();
                for (int k = 2; k <= np; k <<= 1) {
                    for (int j = k >> 1; j > 0; j >>= 1) {
                        for (int t = tid; t < np; t += THREADS) {
                            const int x = t ^ j;
                            if (x > t) {
                                const i64 u = srt[t], v = srt[x];
                                const bool up = (t & k) == 0;
                                if ((u > v) == up) {
                                    srt[t] = v;
                                    srt[x] = u;
                                }
                            }
                        }
                        SYNC();
                    }
                }
                for (int j = tid; j < total; j += THREADS) {
                    const i64 o = srt[j];
                    ST(S_PLAN_HOST, (int *)a.plan_host, prow + j, a.ext_plan, (int)(o >> LOG));
                    ST(S_PLAN_DEV, (int *)a.plan_dev, prow + j, a.ext_plan,
                       LD(S_DEVS_W, devs, mb + (int)(o & (KB - 1)), a.ext_marks, -1));
                }
            }
            filled += total;
        }
        if (HALT(7)) return;
    }
    SYNC();
    TICK(7);
    if (tid == 0) {
        ST(S_PLAN_COUNT, (int *)a.plan_count, lane, a.ext_pcount, (int)filled);
#if PCIE
        atomicAdd((u64 *)a.pcie + a.pcie_rows, (u64)filled);
        if (lane == 0) atomicAdd((u64 *)a.pcie + a.pcie_calls, 1ull);
#endif
    }
}
