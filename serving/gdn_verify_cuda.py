from __future__ import annotations

import hashlib
import os

import torch

ENV = "TAAF_VLLM_GDN_VERIFY_CUDA"
ARCH_ENV = "TAAF_VLLM_GDN_VERIFY_CUDA_ARCH"
DEFAULT_ARCH = "sm_120a"
_BUILT = {}

SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef KD
#define KD 128
#endif
#ifndef VD
#define VD 128
#endif
#ifndef NH
#define NH 16
#endif
#ifndef NHV
#define NHV 48
#endif
#ifndef WIN
#define WIN 5
#endif
#ifndef BVT
#define BVT 64
#endif
#ifndef NACC
#define NACC 8
#endif
#ifndef PF
#define PF 1
#endif

#define PER (KD / BVT)

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float load_param(const void* p, int kind, int i) {
    if (kind == 1) return __bfloat162float(((const bf16*)p)[i]);
    if (kind == 2) return __half2float(((const __half*)p)[i]);
    return ((const float*)p)[i];
}

__device__ __forceinline__ float tree(const float* acc) {
    float s[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) s[i] = acc[i];
#pragma unroll
    for (int w = NACC / 2; w >= 1; w >>= 1) {
#pragma unroll
        for (int i = 0; i < w; ++i) s[i] = s[i] + s[i + w];
    }
    return s[0];
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__global__ void __launch_bounds__(BVT) gdn_verify_ahead_cuda(
    const void* __restrict__ A_log, const bf16* __restrict__ a, const bf16* __restrict__ b,
    const void* __restrict__ dt_bias, const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, bf16* __restrict__ o, int8_t* __restrict__ state,
    const int* __restrict__ cu, const int* __restrict__ sidx, const int* __restrict__ slots,
    const int* __restrict__ pending, const int* __restrict__ phase, float* __restrict__ corr,
    float* __restrict__ keys, float* __restrict__ gates, long long* __restrict__ bad,
    float scale, float sp_beta, float sp_thr, long long sq, long long sk, long long sv, long long sa,
    long long sb, long long so, long long sstate, long long sidx_stride, long long skey_set,
    long long sgate_set, int record_rows, int use_pdl, int pkind) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (use_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
#endif
    const int iv = blockIdx.x;
    const long long nh = blockIdx.y;
    const int n = (int)(nh / NHV);
    const int hv = (int)(nh % NHV);
    const int hk = hv / (NHV / NH);
    const long long bos = cu[n];
    const long long eos = cu[n + 1];
    const long long block = sidx[(long long)n * sidx_stride];
    const long long slot = slots[n];
    const long long length = eos - bos;
    if (length <= 0) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int row = iv * BVT + tid;
    bf16* po = o + bos * so + (long long)hv * VD + row;
    if (block <= 0 || slot < 0 || slot >= record_rows) {
        for (long long t = 0; t < length; ++t) po[t * so] = __float2bfloat16(0.f);
        return;
    }

    __shared__ __align__(16) float sQ[2][KD];
    __shared__ __align__(16) float sK[2][KD];
    __shared__ __align__(16) float sKeys[WIN][KD];

    const long long side = phase[slot];
    int count = pending[slot];
    count = count < 0 ? 0 : (count > WIN ? WIN : count);

    int8_t* base = state + block * sstate;
    const int8_t* codes = base + ((long long)hv * VD + row) * KD;
    float* scl_ptr = (float*)(base + (long long)NHV * VD * KD) + (long long)hv * VD + row;
    const float scl = *scl_ptr;
    if (bad != nullptr && !(fabsf(scl) <= 3.4028234e38f)) atomicAdd((unsigned long long*)bad, 1ULL);

    float h[KD];
#pragma unroll
    for (int c = 0; c < KD / 16; ++c) {
        int4 w = *(const int4*)(codes + c * 16);
        const int8_t* p = (const int8_t*)&w;
#pragma unroll
        for (int j = 0; j < 16; ++j) h[c * 16 + j] = (float)p[j] * scl;
    }

    const float a_log = load_param(A_log, pkind & 3, hv);
    const float bias_t = load_param(dt_bias, (pkind >> 2) & 3, hv);

    if (count > 0) {
        const float* pkeys = keys + side * skey_set + ((slot * NH + hk) * WIN) * KD;
        for (int i = tid; i < count * KD; i += BVT) sKeys[i / KD][i % KD] = pkeys[i];
        const float* pgate = gates + side * sgate_set + (slot * NHV + hv) * WIN;
        const float* pcorr = corr + ((slot * NHV + hv) * WIN) * VD + row;
        float g[WIN], cv[WIN];
#pragma unroll
        for (int t = 0; t < WIN; ++t) {
            g[t] = t < count ? pgate[t] : 0.f;
            cv[t] = t < count ? pcorr[t * VD] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < WIN; ++t) {
            if (t < count) {
                const float e = __expf(g[t]);
#pragma unroll
                for (int j = 0; j < KD; ++j) h[j] = fmaf(cv[t], sKeys[t][j], h[j] * e);
            }
        }
        float mx = 0.f;
#pragma unroll
        for (int j = 0; j < KD; ++j) mx = fmaxf(mx, fabsf(h[j]));
        const float ns = mx > 0.f ? mx / 127.f : 1.f;
        int8_t* wcodes = base + ((long long)hv * VD + row) * KD;
#pragma unroll
        for (int c = 0; c < KD / 16; ++c) {
            int4 w;
            int8_t* p = (int8_t*)&w;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                float x = h[c * 16 + j] / ns;
                x = x >= 0.f ? x + 0.5f : x - 0.5f;
                int ci = (int)x;
                ci = ci > 127 ? 127 : (ci < -128 ? -128 : ci);
                p[j] = (int8_t)ci;
                h[c * 16 + j] = (float)ci * ns;
            }
            *(int4*)(wcodes + c * 16) = w;
        }
        *scl_ptr = ns;
        if (bad != nullptr && !(fabsf(ns) <= 3.4028234e38f)) atomicAdd((unsigned long long*)(bad + 1), 1ULL);
    }

    const bf16* pq = q + bos * sq + (long long)hk * KD;
    const bf16* pk = k + bos * sk + (long long)hk * KD;
    const bf16* pv = v + bos * sv + (long long)hv * VD + row;
    const bf16* pa = a + bos * sa + hv;
    const bf16* pb = b + bos * sb + hv;

    float rq[PER], rk[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        rq[i] = __bfloat162float(pq[tid + i * BVT]);
        rk[i] = __bfloat162float(pk[tid + i * BVT]);
        sQ[0][tid + i * BVT] = rq[i];
        sK[0][tid + i * BVT] = rk[i];
    }
    float nv = __bfloat162float(pv[0]);
    float na = __bfloat162float(pa[0]);
    float nb = __bfloat162float(pb[0]);
    __syncthreads();

    const bool head_writer = iv == 0;
    const bool key_writer = head_writer && (hv % (NHV / NH) == 0);
    float* pcorr_w = corr + ((slot * NHV + hv) * WIN) * VD + row;
    float* pgate_w = gates + (1 - side) * sgate_set + (slot * NHV + hv) * WIN;
    float* pkey_w = keys + (1 - side) * skey_set + ((slot * NH + hk) * WIN) * KD;

    for (long long t = 0; t < length; ++t) {
        const int cur = (int)(t & 1);
        const float cvv = nv, ca = na, cb = nb;
        const bool more = t + 1 < length;
#if PF
        if (more) {
            const long long d = (t + 1);
#pragma unroll
            for (int i = 0; i < PER; ++i) {
                rq[i] = __bfloat162float(pq[d * sq + tid + i * BVT]);
                rk[i] = __bfloat162float(pk[d * sk + tid + i * BVT]);
            }
            nv = __bfloat162float(pv[d * sv]);
            na = __bfloat162float(pa[d * sa]);
            nb = __bfloat162float(pb[d * sb]);
        }
#endif
        float qs = 0.f, ks = 0.f;
#pragma unroll
        for (int i = lane; i < KD; i += 32) {
            const float x = sQ[cur][i], y = sK[cur][i];
            qs = fmaf(x, x, qs);
            ks = fmaf(y, y, ks);
        }
        qs = warp_sum(qs);
        ks = warp_sum(ks);
        const float invq = rsqrtf(qs + 1e-6f);
        const float invk = rsqrtf(ks + 1e-6f);
        const float x_g = ca + bias_t;
        const float sp = (sp_beta * x_g <= sp_thr) ? (1.f / sp_beta) * logf(1.f + expf(sp_beta * x_g)) : x_g;
        const float g = -expf(a_log) * sp;
        const float eg = expf(g);
        const float beta = 1.f / (1.f + expf(-cb));

        float acc[NACC];
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
#pragma unroll
        for (int j = 0; j < KD; j += 4) {
            const float4 kv = *(const float4*)&sK[cur][j];
            acc[(j + 0) % NACC] = fmaf(h[j + 0], kv.x, acc[(j + 0) % NACC]);
            acc[(j + 1) % NACC] = fmaf(h[j + 1], kv.y, acc[(j + 1) % NACC]);
            acc[(j + 2) % NACC] = fmaf(h[j + 2], kv.z, acc[(j + 2) % NACC]);
            acc[(j + 3) % NACC] = fmaf(h[j + 3], kv.w, acc[(j + 3) % NACC]);
        }
        const float S = tree(acc);
        const float bvv = (cvv - eg * S * invk) * beta;
        const float w = bvv * invk;
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
#pragma unroll
        for (int j = 0; j < KD; j += 4) {
            const float4 kv = *(const float4*)&sK[cur][j];
            const float4 qv = *(const float4*)&sQ[cur][j];
            h[j + 0] = fmaf(w, kv.x, h[j + 0] * eg);
            h[j + 1] = fmaf(w, kv.y, h[j + 1] * eg);
            h[j + 2] = fmaf(w, kv.z, h[j + 2] * eg);
            h[j + 3] = fmaf(w, kv.w, h[j + 3] * eg);
            acc[(j + 0) % NACC] = fmaf(h[j + 0], qv.x, acc[(j + 0) % NACC]);
            acc[(j + 1) % NACC] = fmaf(h[j + 1], qv.y, acc[(j + 1) % NACC]);
            acc[(j + 2) % NACC] = fmaf(h[j + 2], qv.z, acc[(j + 2) % NACC]);
            acc[(j + 3) % NACC] = fmaf(h[j + 3], qv.w, acc[(j + 3) % NACC]);
        }
        const float ov = tree(acc) * (invq * scale);
        po[t * so] = __float2bfloat16(ov);
        if (t < WIN) {
            pcorr_w[t * VD] = bvv;
            if (head_writer && tid == 0) pgate_w[t] = g;
            if (key_writer)
                for (int i = tid; i < KD; i += BVT) pkey_w[t * KD + i] = sK[cur][i] * invk;
        }
#if !PF
        if (more) {
            const long long d = (t + 1);
#pragma unroll
            for (int i = 0; i < PER; ++i) {
                rq[i] = __bfloat162float(pq[d * sq + tid + i * BVT]);
                rk[i] = __bfloat162float(pk[d * sk + tid + i * BVT]);
            }
            nv = __bfloat162float(pv[d * sv]);
            na = __bfloat162float(pa[d * sa]);
            nb = __bfloat162float(pb[d * sb]);
        }
#endif
        if (more) {
#pragma unroll
            for (int i = 0; i < PER; ++i) {
                sQ[cur ^ 1][tid + i * BVT] = rq[i];
                sK[cur ^ 1][tid + i * BVT] = rk[i];
            }
        }
        __syncthreads();
    }
}

static int param_kind(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat || t.scalar_type() == torch::kBFloat16 || t.scalar_type() == torch::kHalf,
                name, " must be float32, bfloat16 or float16");
    return t.scalar_type() == torch::kBFloat16 ? 1 : (t.scalar_type() == torch::kHalf ? 2 : 0);
}

void gdn_verify_ahead(torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
                      torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
                      torch::Tensor state, torch::Tensor cu, torch::Tensor sidx, torch::Tensor slots,
                      torch::Tensor pending, torch::Tensor phase, torch::Tensor corr, torch::Tensor keys,
                      torch::Tensor gates, torch::Tensor bad, int64_t rows, double scale, double sp_beta,
                      double sp_thr, int64_t state_stride_bytes, int64_t sidx_stride, int64_t record_rows,
                      bool use_pdl) {
    dim3 grid(VD / BVT, (unsigned)(rows * NHV));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(BVT);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = use_pdl ? 1 : 0;
    long long* badp = bad.numel() ? (long long*)bad.data_ptr() : nullptr;
    cudaLaunchKernelEx(&cfg, gdn_verify_ahead_cuda, (const void*)A_log.data_ptr(),
                       (const bf16*)a.data_ptr(), (const bf16*)b.data_ptr(), (const void*)dt_bias.data_ptr(),
                       (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(), (const bf16*)v.data_ptr(),
                       (bf16*)o.data_ptr(), (int8_t*)state.data_ptr(), (const int*)cu.data_ptr(),
                       (const int*)sidx.data_ptr(), (const int*)slots.data_ptr(), (const int*)pending.data_ptr(),
                       (const int*)phase.data_ptr(), (float*)corr.data_ptr(), (float*)keys.data_ptr(),
                       (float*)gates.data_ptr(), badp, (float)scale, (float)sp_beta, (float)sp_thr,
                       (long long)q.stride(0), (long long)k.stride(0), (long long)v.stride(0),
                       (long long)a.stride(0), (long long)b.stride(0), (long long)o.stride(0),
                       (long long)state_stride_bytes, (long long)sidx_stride, (long long)keys.stride(0),
                       (long long)gates.stride(0), (int)record_rows, (int)use_pdl,
                       param_kind(A_log, "A_log") | (param_kind(dt_bias, "dt_bias") << 2));
}
"""

SOURCE_LR = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef KD
#define KD 128
#endif
#ifndef VD
#define VD 128
#endif
#ifndef NH
#define NH 16
#endif
#ifndef NHV
#define NHV 48
#endif
#ifndef WIN
#define WIN 5
#endif
#ifndef BVT
#define BVT 128
#endif
#ifndef MINB
#define MINB 0
#endif
#ifndef LDGH
#define LDGH 0
#endif
#ifndef COHERENT
#define COHERENT 0
#endif
#if COHERENT
#define RIN
#else
#define RIN __restrict__
#endif

#define STG (LDGH == 2)
#define NW (BVT / 32)
#define TILES (KD / 32)
#define NDOT 35
#define COLS 12
#define MAGIC 12582912.0f

#define O_EG 0
#define O_BETA 5
#define O_G 10
#define O_FE 15
#define O_IK 20
#define O_OS 25
#define O_P 30
#define O_CK 35
#define O_CQ 60
#define O_PF 85
#define O_FF 86
#define TBL 96

static_assert(WIN <= 5, "the low-rank verify keeps at most five tokens");
static_assert(KD == 128 && VD % BVT == 0 && BVT >= 64 && BVT <= 128, "tile shape");

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float load_param(const void* p, int kind, int i) {
    if (kind == 1) return __bfloat162float(((const bf16*)p)[i]);
    if (kind == 2) return __half2float(((const __half*)p)[i]);
    return ((const float*)p)[i];
}

#define PAIRS_A(X) X(0,0) X(1,1) X(2,2) X(3,3) X(4,4) X(5,5) X(6,6) X(7,7) X(8,8) X(9,9)
#define PAIRS_B(X) X(0,1) X(0,2) X(0,3) X(0,4) X(1,2) X(1,3) X(1,4) X(2,3) X(2,4) X(3,4)
#define PAIRS_C(X) X(0,5) X(0,6) X(1,6) X(0,7) X(1,7) X(2,7) X(0,8) X(1,8) X(2,8) X(3,8) X(0,9) X(1,9) X(2,9) X(3,9) X(4,9)

struct Acc {
    float k0, k1, k2, k3, k4, q0, q1, q2, q3, q4;
};

__device__ __forceinline__ int code_of(unsigned w, int pos) {
    return (int)(int8_t)(w >> (8 * pos));
}

__device__ __forceinline__ void dots(Acc& a, float r, const float* kq) {
    const float4 c0 = *(const float4*)(kq);
    const float4 c1 = *(const float4*)(kq + 4);
    const float4 c2 = *(const float4*)(kq + 8);
    a.k0 = fmaf(r, c0.x, a.k0);
    a.k1 = fmaf(r, c0.y, a.k1);
    a.k2 = fmaf(r, c0.z, a.k2);
    a.k3 = fmaf(r, c0.w, a.k3);
    a.k4 = fmaf(r, c1.x, a.k4);
    a.q0 = fmaf(r, c1.y, a.q0);
    a.q1 = fmaf(r, c1.z, a.q1);
    a.q2 = fmaf(r, c1.w, a.q2);
    a.q3 = fmaf(r, c2.x, a.q3);
    a.q4 = fmaf(r, c2.y, a.q4);
}

__device__ __forceinline__ float fold_value(float p0, unsigned word, int pos, const float* cs, const float* key) {
    const float4 ka = *(const float4*)(key);
    const float kb = key[4];
    float x = fmaf(p0, (float)code_of(word, pos), cs[0] * ka.x);
    x = fmaf(cs[1], ka.y, x);
    x = fmaf(cs[2], ka.z, x);
    x = fmaf(cs[3], ka.w, x);
    x = fmaf(cs[4], kb, x);
    return x;
}

__device__ __forceinline__ void load_cols(const float* p, float (&r)[COLS]) {
#pragma unroll
    for (int c = 0; c < COLS / 4; ++c) {
        const float4 x = *(const float4*)(p + 4 * c);
        r[4 * c + 0] = x.x;
        r[4 * c + 1] = x.y;
        r[4 * c + 2] = x.z;
        r[4 * c + 3] = x.w;
    }
}

template <int T>
__device__ __forceinline__ void token_step(const float* sT, float scale_row, float hk, float hq, float vt,
                                           float (&bb)[5], bf16* po_t, float* pc_t) {
    float u = (T == 0 ? 1.f : sT[O_P + (T == 0 ? 0 : T - 1)]) * (scale_row * hk * sT[O_IK + T]);
#pragma unroll
    for (int s = 0; s < T; ++s) u = fmaf(sT[O_CK + T * 5 + s], bb[s], u);
    const float bt = (vt - sT[O_EG + T] * u) * sT[O_BETA + T];
    bb[T] = bt;
    float oq = sT[O_P + T] * (scale_row * hq);
#pragma unroll
    for (int s = 0; s <= T; ++s) oq = fmaf(sT[O_CQ + T * 5 + s], bb[s], oq);
    *po_t = __float2bfloat16(oq * sT[O_OS + T]);
    *pc_t = bt;
}

__device__ __forceinline__ float warp_xor_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 16);
    x += __shfl_xor_sync(0xffffffffu, x, 8);
    x += __shfl_xor_sync(0xffffffffu, x, 4);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    return x;
}

__device__ __forceinline__ uint4 chunk_in(const int8_t* rowp, const uint4* srow, int c) {
#if STG
    return srow[c];
#elif LDGH == 1
    return __ldcs((const uint4*)(rowp + 16 * c));
#else
    return *(const uint4*)(rowp + 16 * c);
#endif
}

#if MINB > 0
__global__ void __launch_bounds__(BVT, MINB) gdn_verify_ahead_lr(
#else
__global__ void __launch_bounds__(BVT) gdn_verify_ahead_lr(
#endif
    const void* RIN A_log, const bf16* RIN a, const bf16* RIN b,
    const void* RIN dt_bias, const bf16* RIN q, const bf16* RIN k,
    const bf16* RIN v, bf16* __restrict__ o, int8_t* __restrict__ state,
    const int* RIN cu, const int* RIN sidx, const int* RIN slots,
    const int* RIN pending, const int* RIN phase, float* __restrict__ corr,
    float* __restrict__ keys, float* __restrict__ gates, long long* __restrict__ bad,
    float scale, float sp_beta, float sp_thr, long long sq, long long sk, long long sv, long long sa,
    long long sb, long long so, long long sstate, long long sidx_stride, long long skey_set,
    long long sgate_set, int record_rows, int use_pdl, int pkind) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (use_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
#endif
    const int iv = blockIdx.x;
    const long long nh = blockIdx.y;
    const int n = (int)(nh / NHV);
    const int hv = (int)(nh % NHV);
    const int hk = hv / (NHV / NH);
    const long long bos = cu[n];
    const long long eos = cu[n + 1];
    const long long block = sidx[(long long)n * sidx_stride];
    const long long slot = slots[n];
    const int length = (int)(eos - bos);
    if (length <= 0) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = iv * BVT + tid;
    bf16* po = o + bos * so + (long long)hv * VD + row;
    if (block <= 0 || slot < 0 || slot >= record_rows) {
        for (int t = 0; t < length; ++t) po[(long long)t * so] = __float2bfloat16(0.f);
        return;
    }
    if (length > WIN) {
        for (int t = 0; t < length; ++t) po[(long long)t * so] = __float2bfloat16(__int_as_float(0x7fc00000));
        return;
    }

    __shared__ __align__(16) float sKQ[KD][COLS];
    __shared__ __align__(16) float sKey[KD][8];
    __shared__ float sPart[TILES][NDOT + 1];
    __shared__ float sDot[NDOT + 1];
    __shared__ float sT[TBL];
#if STG
    __shared__ __align__(16) uint4 sC[BVT * 9];
#endif

    const long long side = phase[slot];
    int count = pending[slot];
    count = count < 0 ? 0 : (count > WIN ? WIN : count);

    int8_t* base = state + block * sstate;
    int8_t* tile_rows = base + ((long long)hv * VD + (long long)iv * BVT) * KD;
    int8_t* rowp = tile_rows + (long long)tid * KD;
    float* scl_ptr = (float*)(base + (long long)NHV * VD * KD) + (long long)hv * VD + row;
#if STG
    uint4* srow = sC + tid * 9;
    for (int idx = tid; idx < BVT * 8; idx += BVT) sC[(idx >> 3) * 9 + (idx & 7)] = *(const uint4*)(tile_rows + (long long)idx * 16);
#else
    const uint4* srow = nullptr;
#endif
    const float scl = *scl_ptr;
    if (bad != nullptr && !(fabsf(scl) <= 3.4028234e38f)) atomicAdd((unsigned long long*)bad, 1ULL);

    const bf16* pq = q + bos * sq + (long long)hk * KD;
    const bf16* pk = k + bos * sk + (long long)hk * KD;
    const bf16* pv = v + bos * sv + (long long)hv * VD + row;
    const float* pkeys = keys + side * skey_set + ((slot * NH + hk) * WIN) * KD;
    const float* pgate = gates + side * sgate_set + (slot * NHV + hv) * WIN;
    const float* pcorr = corr + ((slot * NHV + hv) * WIN) * VD + row;

    float vv[5], cv[5];
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        vv[t] = (t < WIN && t < length) ? __bfloat162float(pv[(long long)t * sv]) : 0.f;
        cv[t] = (t < WIN && t < count) ? pcorr[t * VD] : 0.f;
    }

    for (int i = tid; i < KD; i += BVT) {
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            sKQ[i][t] = (t < WIN && t < length) ? __bfloat162float(pk[(long long)t * sk + i]) : 0.f;
            sKQ[i][5 + t] = (t < WIN && t < length) ? __bfloat162float(pq[(long long)t * sq + i]) : 0.f;
        }
        sKQ[i][10] = 0.f;
        sKQ[i][11] = 0.f;
#pragma unroll
        for (int s = 0; s < 8; ++s) sKey[i][s] = (s < WIN && s < count) ? pkeys[s * KD + i] : 0.f;
    }
    if (tid < 5) {
        const int t = tid;
        float eg = 1.f, beta = 0.f, g = 0.f;
        if (t < WIN && t < length) {
            const float x_g = __bfloat162float(a[(bos + t) * sa + hv]) + load_param(dt_bias, (pkind >> 2) & 3, hv);
            const float sp = (sp_beta * x_g <= sp_thr) ? (1.f / sp_beta) * logf(1.f + expf(sp_beta * x_g)) : x_g;
            g = -expf(load_param(A_log, pkind & 3, hv)) * sp;
            eg = expf(g);
            beta = 1.f / (1.f + expf(-__bfloat162float(b[(bos + t) * sb + hv])));
        }
        sT[O_EG + t] = eg;
        sT[O_BETA + t] = beta;
        sT[O_G + t] = g;
        sT[O_FE + t] = (t < WIN && t < count) ? expf(pgate[t]) : 1.f;
    }
    __syncthreads();

    for (int tile = warp; tile < TILES; tile += NW) {
        const int j = tile * 32 + lane;
#define PROD(A, B) pr[d++] = r[A] * r[B];
        {
            float r[COLS];
            load_cols(&sKQ[j][0], r);
            float pr[10];
            int d = 0;
            PAIRS_A(PROD)
#pragma unroll
            for (int e = 0; e < 10; ++e) pr[e] = warp_xor_sum(pr[e]);
            if (lane == 0) {
#pragma unroll
                for (int e = 0; e < 10; ++e) sPart[tile][e] = pr[e];
            }
        }
        {
            float r[COLS];
            load_cols(&sKQ[j][0], r);
            float pr[10];
            int d = 0;
            PAIRS_B(PROD)
#pragma unroll
            for (int e = 0; e < 10; ++e) pr[e] = warp_xor_sum(pr[e]);
            if (lane == 0) {
#pragma unroll
                for (int e = 0; e < 10; ++e) sPart[tile][10 + e] = pr[e];
            }
        }
        {
            float r[COLS];
            load_cols(&sKQ[j][0], r);
            float pr[15];
            int d = 0;
            PAIRS_C(PROD)
#pragma unroll
            for (int e = 0; e < 15; ++e) pr[e] = warp_xor_sum(pr[e]);
            if (lane == 0) {
#pragma unroll
                for (int e = 0; e < 15; ++e) sPart[tile][20 + e] = pr[e];
            }
        }
#undef PROD
    }
    __syncthreads();

    if (tid < NDOT) {
        float s = sPart[0][tid];
        for (int tile = 1; tile < TILES; ++tile) s += sPart[tile][tid];
        sDot[tid] = s;
    }
    __syncthreads();

    if (tid < 5) {
        const int t = tid;
        const bool live = t < WIN && t < length;
        sT[O_IK + t] = live ? rsqrtf(sDot[t] + 1e-6f) : 1.f;
        sT[O_OS + t] = (live ? rsqrtf(sDot[5 + t] + 1e-6f) : 0.f) * scale;
        float p = 1.f;
        for (int u = 0; u <= t; ++u) p *= sT[O_EG + u];
        sT[O_P + t] = p;
        float f = 1.f;
        for (int u = t + 1; u < 5; ++u) f *= sT[O_FE + u];
        sT[O_FF + t] = f;
    } else if (tid == 5) {
        float pf = 1.f;
        for (int u = 0; u < 5; ++u) pf *= sT[O_FE + u];
        sT[O_PF] = pf;
    }
    __syncthreads();

    if (tid < 25) {
        const int t = tid / 5;
        const int s = tid % 5;
        float ck = 0.f, cq = 0.f;
        if (s < t) {
            float f = 1.f;
            for (int u = s + 1; u <= t - 1; ++u) f *= sT[O_EG + u];
            ck = f * sDot[10 + s * 5 - (s * (s + 1)) / 2 + (t - s - 1)] * sT[O_IK + s] * sT[O_IK + t];
        }
        if (s <= t) {
            float f = 1.f;
            for (int u = s + 1; u <= t; ++u) f *= sT[O_EG + u];
            cq = f * sDot[20 + (t * (t + 1)) / 2 + s] * sT[O_IK + s];
        }
        sT[O_CK + t * 5 + s] = ck;
        sT[O_CQ + t * 5 + s] = cq;
    }
    __syncthreads();

    Acc acc;
    acc.k0 = acc.k1 = acc.k2 = acc.k3 = acc.k4 = 0.f;
    acc.q0 = acc.q1 = acc.q2 = acc.q3 = acc.q4 = 0.f;
    float scale_row = scl;

    if (count > 0) {
        const float p0 = sT[O_PF] * scl;
        float cs[5];
#pragma unroll
        for (int s = 0; s < 5; ++s) cs[s] = sT[O_FF + s] * cv[s];
        float mx = 0.f;
        {
            uint4 nxt = chunk_in(rowp, srow, 0);
#pragma unroll 1
            for (int c = 0; c < KD / 16; ++c) {
                const uint4 w = nxt;
                if (c + 1 < KD / 16) nxt = chunk_in(rowp, srow, c + 1);
                const unsigned wd[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const float x = fold_value(p0, wd[e >> 2], e & 3, cs, &sKey[16 * c + e][0]);
                    mx = fmaxf(mx, fabsf(x));
                }
            }
        }
        const float ns = mx > 0.f ? mx / 127.f : 1.f;
        const float rinv = 1.f / ns;
        {
            uint4 nxt = chunk_in(rowp, srow, 0);
#pragma unroll 1
            for (int c = 0; c < KD / 16; ++c) {
                const uint4 w = nxt;
                if (c + 1 < KD / 16) nxt = chunk_in(rowp, srow, c + 1);
                const unsigned wd[4] = {w.x, w.y, w.z, w.w};
                unsigned pw[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    unsigned packed = 0u;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int j = 16 * c + 4 * g + e;
                        const float x = fold_value(p0, wd[g], e, cs, &sKey[j][0]);
                        const float tv = fmaf(x, rinv, MAGIC);
                        packed |= (__float_as_uint(tv) & 0xFFu) << (8 * e);
                        dots(acc, tv - MAGIC, &sKQ[j][0]);
                    }
                    pw[g] = packed;
                }
                const uint4 out = make_uint4(pw[0], pw[1], pw[2], pw[3]);
#if STG
                srow[c] = out;
#else
                *(uint4*)(rowp + 16 * c) = out;
#endif
            }
        }
#if STG
        __syncthreads();
        for (int idx = tid; idx < BVT * 8; idx += BVT) *(uint4*)(tile_rows + (long long)idx * 16) = sC[(idx >> 3) * 9 + (idx & 7)];
#endif
        *scl_ptr = ns;
        if (bad != nullptr && !(fabsf(ns) <= 3.4028234e38f)) atomicAdd((unsigned long long*)(bad + 1), 1ULL);
        scale_row = ns;
    } else {
        uint4 nxt = chunk_in(rowp, srow, 0);
#pragma unroll 1
        for (int c = 0; c < KD / 16; ++c) {
            const uint4 w = nxt;
            if (c + 1 < KD / 16) nxt = chunk_in(rowp, srow, c + 1);
            const unsigned wd[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int e = 0; e < 16; ++e) dots(acc, (float)code_of(wd[e >> 2], e & 3), &sKQ[16 * c + e][0]);
        }
    }

    float* pcorr_w = corr + ((slot * NHV + hv) * WIN) * VD + row;
    float bb[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    if (0 < WIN && 0 < length) token_step<0>(sT, scale_row, acc.k0, acc.q0, vv[0], bb, po, pcorr_w);
    if (1 < WIN && 1 < length) token_step<1>(sT, scale_row, acc.k1, acc.q1, vv[1], bb, po + so, pcorr_w + VD);
    if (2 < WIN && 2 < length) token_step<2>(sT, scale_row, acc.k2, acc.q2, vv[2], bb, po + 2 * so, pcorr_w + 2 * VD);
    if (3 < WIN && 3 < length) token_step<3>(sT, scale_row, acc.k3, acc.q3, vv[3], bb, po + 3 * so, pcorr_w + 3 * VD);
    if (4 < WIN && 4 < length) token_step<4>(sT, scale_row, acc.k4, acc.q4, vv[4], bb, po + 4 * so, pcorr_w + 4 * VD);

    if (iv == 0) {
        float* pgate_w = gates + (1 - side) * sgate_set + (slot * NHV + hv) * WIN;
        if (tid < WIN && tid < length) pgate_w[tid] = sT[O_G + tid];
        if (hv % (NHV / NH) == 0) {
            float* pkey_w = keys + (1 - side) * skey_set + ((slot * NH + hk) * WIN) * KD;
            for (int t = 0; t < WIN; ++t) {
                if (t < length) {
                    const float ikt = sT[O_IK + t];
                    for (int i = tid; i < KD; i += BVT) pkey_w[t * KD + i] = sKQ[i][t] * ikt;
                }
            }
        }
    }
}

static int param_kind(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat || t.scalar_type() == torch::kBFloat16 || t.scalar_type() == torch::kHalf,
                name, " must be float32, bfloat16 or float16");
    return t.scalar_type() == torch::kBFloat16 ? 1 : (t.scalar_type() == torch::kHalf ? 2 : 0);
}

void gdn_verify_ahead(torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
                      torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
                      torch::Tensor state, torch::Tensor cu, torch::Tensor sidx, torch::Tensor slots,
                      torch::Tensor pending, torch::Tensor phase, torch::Tensor corr, torch::Tensor keys,
                      torch::Tensor gates, torch::Tensor bad, int64_t rows, double scale, double sp_beta,
                      double sp_thr, int64_t state_stride_bytes, int64_t sidx_stride, int64_t record_rows,
                      bool use_pdl) {
    dim3 grid(VD / BVT, (unsigned)(rows * NHV));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(BVT);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = use_pdl ? 1 : 0;
    long long* badp = bad.numel() ? (long long*)bad.data_ptr() : nullptr;
    cudaLaunchKernelEx(&cfg, gdn_verify_ahead_lr, (const void*)A_log.data_ptr(),
                       (const bf16*)a.data_ptr(), (const bf16*)b.data_ptr(), (const void*)dt_bias.data_ptr(),
                       (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(), (const bf16*)v.data_ptr(),
                       (bf16*)o.data_ptr(), (int8_t*)state.data_ptr(), (const int*)cu.data_ptr(),
                       (const int*)sidx.data_ptr(), (const int*)slots.data_ptr(), (const int*)pending.data_ptr(),
                       (const int*)phase.data_ptr(), (float*)corr.data_ptr(), (float*)keys.data_ptr(),
                       (float*)gates.data_ptr(), badp, (float)scale, (float)sp_beta, (float)sp_thr,
                       (long long)q.stride(0), (long long)k.stride(0), (long long)v.stride(0),
                       (long long)a.stride(0), (long long)b.stride(0), (long long)o.stride(0),
                       (long long)state_stride_bytes, (long long)sidx_stride, (long long)keys.stride(0),
                       (long long)gates.stride(0), (int)record_rows, (int)use_pdl,
                       param_kind(A_log, "A_log") | (param_kind(dt_bias, "dt_bias") << 2));
}
"""

CPP = """
void gdn_verify_ahead(torch::Tensor A_log, torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
                      torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
                      torch::Tensor state, torch::Tensor cu, torch::Tensor sidx, torch::Tensor slots,
                      torch::Tensor pending, torch::Tensor phase, torch::Tensor corr, torch::Tensor keys,
                      torch::Tensor gates, torch::Tensor bad, int64_t rows, double scale, double sp_beta,
                      double sp_thr, int64_t state_stride_bytes, int64_t sidx_stride, int64_t record_rows,
                      bool use_pdl);
"""

DEFAULT_VARIANT = (128, 0, 0, 1)
LEGACY_VARIANT = (64, 8, 1)
PTXAS_ENV = "TAAF_VLLM_GDN_VERIFY_CUDA_PTXAS"
COHERENT_ENV = "TAAF_VLLM_GDN_VERIFY_CUDA_COHERENT"


def parse(said):
    said = str(said or "").strip()
    if not said or said == "0":
        return None
    if said == "1":
        return DEFAULT_VARIANT
    try:
        variant = tuple(int(part) for part in said.split(","))
    except ValueError:
        variant = ()
    if len(variant) == 4 and variant[3] == 1:
        if variant[0] in (64, 128) and variant[1] in (0, 4, 6, 8, 12, 16) and variant[2] in (0, 1, 2):
            return variant
    elif len(variant) in (3, 4) and (len(variant) == 3 or variant[3] == 0):
        if variant[0] in (32, 64, 128) and variant[1] in (2, 4, 8) and variant[2] in (0, 1):
            return variant[:3]
    raise ValueError(f"{ENV} must be 0, 1, bv,accumulators,prefetch (bv 32/64/128, accumulators 2/4/8, prefetch 0/1) "
                     f"or bv,min_blocks,load,1 for the low-register kernel (bv 64/128, min_blocks 0 for none "
                     f"or 4/6/8/12/16, load 0 direct, 1 streaming, 2 staged through shared memory), found {said!r}")


def variant():
    return parse(os.environ.get(ENV))


def coherent(variant=None):
    said = str(os.environ.get(COHERENT_ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise ValueError(f"{COHERENT_ENV} must be '', '0' or '1', found {said!r}")
    if said == "1" and variant is not None and len(variant) != 4:
        raise ValueError(f"{COHERENT_ENV}=1 shapes the low-register kernel (bv,min_blocks,load,1), found the earlier "
                         f"variant {','.join(str(part) for part in variant)}")
    return said == "1"


def build(variant, heads, hv_heads, k_dim, v_dim, window):
    plain = coherent(variant)
    key = (tuple(variant), heads, hv_heads, k_dim, v_dim, window, plain)
    held = _BUILT.get(key)
    if held is not None:
        return held
    from torch.utils.cpp_extension import load_inline
    low_register = len(variant) == 4
    bv, second, third = variant[:3]
    common = [f"-DKD={k_dim}", f"-DVD={v_dim}", f"-DNH={heads}", f"-DNHV={hv_heads}", f"-DWIN={window}",
              f"-DBVT={bv}", "-O3", "--use_fast_math", "-lineinfo", f"-arch={os.environ.get(ARCH_ENV, DEFAULT_ARCH)}"]
    verbose = os.environ.get(PTXAS_ENV, "") == "1"
    if verbose:
        common.append("-Xptxas=-v")
    if low_register:
        flags, source = common + [f"-DMINB={second}", f"-DLDGH={third}"] + (["-DCOHERENT=1"] if plain else []), SOURCE_LR
    else:
        flags, source = common + [f"-DNACC={second}", f"-DPF={third}"], SOURCE
    source = source.replace("", "")
    tag = hashlib.sha1(("|".join(flags) + source).encode()).hexdigest()[:12]
    held = load_inline(name=f"gdn_verify_cuda_{tag}", cpp_sources=[CPP], cuda_sources=[source],
                       functions=["gdn_verify_ahead"], extra_cuda_cflags=flags, verbose=verbose)
    _BUILT[key] = held
    if plain:
        print(f"GDN_VERIFY_CUDA_COHERENT built {','.join(str(part) for part in variant)}: the low-register verify reads "
              f"q, k, v, a, b, the gate parameters and the indices through coherent global loads instead of the "
              f"read-only cache ({COHERENT_ENV}=1)", flush=True)
    return held


PARAM_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _describe(tensor):
    return f"dtype={str(tensor.dtype).replace('torch.', '')} shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}"


def refusals(*, state, q, k, v, a, b, o, A_log, dt_bias, cu, sidx, slots, pending, phase, correction, key_sets,
             gate_sets, k_dim, v_dim, bv, low_register=False):
    found = []
    if v_dim % bv or k_dim % 16 or k_dim % bv:
        found.append(f"geometry: block_v {bv} needs v_dim {v_dim} and k_dim {k_dim} to divide by it and k_dim by 16")
    if low_register and k_dim != 128:
        found.append(f"geometry: the low-register kernel needs k_dim 128, found {k_dim}")
    if state.dim() != 4 or tuple(state.stride()[1:]) != (v_dim * k_dim, k_dim, 1):
        found.append(f"int8 state: needs [blocks, heads, {v_dim}, {k_dim}] with contiguous heads, found {_describe(state)}")
    else:
        need = state.shape[1] * v_dim * (k_dim + 4)
        if state.stride(0) * state.element_size() < need:
            found.append(f"int8 state: a block holds {state.stride(0) * state.element_size()} bytes, the codes and "
                         f"row scales need {need}")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b), ("o", o)):
        if tensor.dtype != torch.bfloat16 or tensor.stride(-1) != 1:
            found.append(f"bf16 inputs: {name} needs bfloat16 with unit stride on its last axis, found {_describe(tensor)}")
    row_bytes = state.stride(0) * state.element_size()
    if state.data_ptr() % 16 or row_bytes % 16:
        found.append(f"16-byte rows: state pointer % 16 = {state.data_ptr() % 16} and block stride {row_bytes} bytes "
                     f"% 16 = {row_bytes % 16}, both must be 0")
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        if tensor.dtype not in PARAM_DTYPES:
            found.append(f"params: {name} needs float32, bfloat16 or float16, found {_describe(tensor)}")
    for name, tensor in (("correction", correction), ("key_sets", key_sets), ("gate_sets", gate_sets)):
        if tensor.dtype != torch.float32:
            found.append(f"records: {name} needs float32, found {_describe(tensor)}")
    for name, tensor in (("cu", cu), ("sidx", sidx), ("slots", slots), ("pending", pending), ("phase", phase)):
        if tensor.dtype != torch.int32:
            found.append(f"indices: {name} needs int32, found {_describe(tensor)}")
    for name, tensor in (("cu", cu), ("slots", slots), ("pending", pending), ("phase", phase)):
        if tensor.dim() != 1 or tensor.stride(0) != 1:
            found.append(f"indices: {name} needs a unit stride, found {_describe(tensor)}")
    return found


def launch(module, *, A_log, a, b, dt_bias, q, k, v, o, state, cu, sidx, slots, pending, phase, correction,
           key_sets, gate_sets, rows, scale, softplus_beta, softplus_threshold, record_rows, pdl, bad=None):
    module.gdn_verify_ahead(
        A_log, a, b, dt_bias, q, k, v, o, state, cu, sidx,
        slots, pending, phase, correction, key_sets, gate_sets,
        bad if bad is not None else torch.empty(0, dtype=torch.int64, device=q.device),
        rows, scale, softplus_beta, softplus_threshold, state.stride(0) * state.element_size(),
        sidx.stride(0), record_rows, bool(pdl))
