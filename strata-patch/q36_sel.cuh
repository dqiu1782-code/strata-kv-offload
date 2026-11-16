// q36_sel.cuh - training-free block selection for the decode attention of the Qwen3.6 engine (included by
// q36_engine.cu inside its anonymous namespace, after block_sum / block_max / sigm).
//
// The idea (kv-offload project): a decode step does not read every cell of the context but 96 blocks of 64 cells: block 0,
// the 8 newest blocks, and the best of the rest by a score on the keys, per KV head:
//   * every 16 cells a window of 32 keys is averaged (km: [kv head][window][256]);
//   * the query's softmax weight on each window (summed over the 8 query heads of the KV head), max-pooled over the 5
//     windows that overlap a block, is the block's score (the InfLLM-v2 / MiniCPM4 style score).
// Enabled with STRATA_Q36_SEL=1.  Everything runs on the device from the position scalar, so it sits inside the decode graph.
// When the context has 96 blocks or fewer the selection is all of them in order, and k_sel_part / k_sel_merge then cover
// the same cells per chunk as k_attn_part / k_attn_merge (the sums run in another order).
#pragma once

#include <cstdint>

constexpr int kSelBS = 64;       // cells per block
constexpr int kSelWin = 32;      // cells per key window
constexpr int kSelStride = 16;   // window stride
constexpr int kSelN = 96;        // blocks read per step
constexpr int kSelLocal = 8;     // newest blocks always read
constexpr int kSelNbMax = 4096;  // most blocks (a context of 262144)
constexpr int kSelG = 8;         // query heads per KV head
constexpr int kSelHD = 256;      // head dim

__host__ __device__ __forceinline__ int sel_windows(int L) { return L >= kSelWin ? (L - kSelWin) / kSelStride + 1 : 0; }

__device__ __forceinline__ float sel_wsum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    return x;
}

/// Window means of the keys just written: T cells from *pos0.  grid (T / 16 + 2, HKV), block 256.
__global__ void k_kmean_upd(const float* kc, float* km, int ctx, int nkmax, const int32_t* pos0, int T) {
    const int p0 = *pos0;
    const int w = sel_windows(p0) + (int) blockIdx.x;
    if (w >= sel_windows(p0 + T)) return;
    const int kh = blockIdx.y, d = threadIdx.x;
    const float* src = kc + ((size_t) kh * ctx + (size_t) w * kSelStride) * kSelHD + d;
    float s = 0.f;
    for (int j = 0; j < kSelWin; ++j) s += src[(size_t) j * kSelHD];
    km[((size_t) kh * nkmax + w) * kSelHD + d] = s * (1.0f / kSelWin);
}

/// logit[kh][g][w] = q[kh * 8 + g] . km[kh][w] / 16 for one decode token.  grid (ceil(nkmax / 8), HKV), block 256: one
/// warp per window.
__global__ void k_sel_score(const float* q, const float* km, float* logit, int nkmax, const int32_t* pos0) {
    __shared__ float qs[kSelG * kSelHD];
    const int kh = blockIdx.y;
    const int nk = sel_windows(*pos0 + 1);
    for (int i = threadIdx.x; i < kSelG * kSelHD; i += blockDim.x) qs[i] = q[(size_t) kh * kSelG * kSelHD + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int w = (int) blockIdx.x * 8 + warp;
    if (w >= nk) return;
    const float4* kp = reinterpret_cast<const float4*>(km + ((size_t) kh * nkmax + w) * kSelHD);
    const float4 a = kp[lane], b = kp[lane + 32];
    for (int g = 0; g < kSelG; ++g) {
        const float* qg = qs + g * kSelHD;
        float s = qg[lane * 4] * a.x + qg[lane * 4 + 1] * a.y + qg[lane * 4 + 2] * a.z + qg[lane * 4 + 3] * a.w +
                  qg[128 + lane * 4] * b.x + qg[128 + lane * 4 + 1] * b.y + qg[128 + lane * 4 + 2] * b.z +
                  qg[128 + lane * 4 + 3] * b.w;
        s = sel_wsum(s);
        if (lane == 0) logit[((size_t) kh * kSelG + g) * nkmax + w] = s * 0.0625f;
    }
}

/// The 96 blocks of each KV head, ascending, -1 padded.  grid HKV, block 1024.
__global__ void k_sel_pick(const float* logit, int nkmax, const int32_t* pos0, int32_t* ids) {
    __shared__ uint32_t key[kSelNbMax];
    __shared__ float red[1024];
    __shared__ float mg[kSelG], ig[kSelG];
    __shared__ int sel[kSelN];
    __shared__ int cnt;
    const int kh = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int L = *pos0 + 1, nb = (L + kSelBS - 1) / kSelBS, nk = sel_windows(L);
    int32_t* out = ids + kh * kSelN;
    if (nb <= kSelN || nk < 1 || nb > kSelNbMax) {   // short context: every block
        for (int i = tid; i < kSelN; i += nt) out[i] = i < nb ? i : -1;
        return;
    }
    const float* lg = logit + (size_t) kh * kSelG * nkmax;
    for (int g = 0; g < kSelG; ++g) {   // softmax statistics over the windows of query head g
        float m = -INFINITY;
        for (int w = tid; w < nk; w += nt) m = fmaxf(m, lg[(size_t) g * nkmax + w]);
        red[tid] = m;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        const float mx = red[0];
        __syncthreads();
        float sum = 0.f;
        for (int w = tid; w < nk; w += nt) sum += expf(lg[(size_t) g * nkmax + w] - mx);
        red[tid] = sum;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            mg[g] = mx;
            ig[g] = 1.0f / red[0];
        }
        __syncthreads();
    }
    const int qi = (L - 1) / kSelBS;
    for (int b = tid; b < nb; b += nt) {   // block score: the best window among the five that overlap it
        float best = -INFINITY;
        const int lo = max(4 * b - 1, 0), hi = min(4 * b + 3, nk - 1);
        for (int w = lo; w <= hi; ++w) {
            float sv = 0.f;
            for (int g = 0; g < kSelG; ++g) sv += expf(lg[(size_t) g * nkmax + w] - mg[g]) * ig[g];
            best = fmaxf(best, sv);
        }
        if (b == 0 || (b >= max(qi - kSelLocal + 1, 0) && b <= qi)) best = INFINITY;
        const uint32_t u = __float_as_uint(best);
        key[b] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // order-preserving
    }
    __syncthreads();
    // the largest key T with at least kSelN keys >= T (bitwise search; hardware vote counts)
    uint32_t T = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = T | (1u << bit);
        int c = 0;
        for (int base = 0; base < nb; base += nt) {
            const int b = base + tid;
            c += __syncthreads_count(b < nb && key[b] >= cand);
        }
        if (c >= kSelN) T = cand;
    }
    if (tid == 0) cnt = 0;
    __syncthreads();
    int ties = 0;
    for (int base = 0; base < nb; base += nt) {   // every key above T, and how many equal T
        const int b = base + tid;
        const bool gt = b < nb && key[b] > T;
        ties += __syncthreads_count(b < nb && key[b] == T);
        if (gt) sel[atomicAdd(&cnt, 1)] = b;
    }
    __syncthreads();
    const int need = kSelN - cnt;
    if (ties == need) {   // the usual case: all the ties are in
        for (int b = tid; b < nb; b += nt)
            if (key[b] == T) sel[atomicAdd(&cnt, 1)] = b;
    } else if (tid == 0) {   // more ties than room: the lowest ids
        int n = cnt, left = need;
        for (int b = 0; b < nb && left > 0; ++b)
            if (key[b] == T) {
                sel[n++] = b;
                --left;
            }
    }
    __syncthreads();
    if (tid < kSelN) {   // ascending order
        const int v = sel[tid];
        int r = 0;
        for (int j = 0; j < kSelN; ++j) r += sel[j] < v;
        out[r] = v;
    }
}

/// The block pool (K / V of the 10 attention layers live in pinned host memory; the GPU keeps kP slots of one 64-cell block
/// per KV head and layer, an LRU cache).  A slot code is the slot, plus kPoolMiss when the block has to be copied in this
/// step; -1 means "read the block straight from host memory" (the newest blocks, which are still being written).
constexpr int kPoolMiss = 0x40000000;
constexpr int kPoolDirect = 9;   // blocks within this many of the newest are never cached
constexpr int kPoolRing = 16;    // ... they live in a GPU ring of the newest 16 blocks (pool slots P .. P + 15)

/// Mirrors the cells [pos0 + T - n, pos0 + T) just written to host memory (n = min(T, 1024)) into the ring slots P + (block
/// & 15).  grid (n, HKV), block 256.  P is the number of cache slots; the slot stride is P + kPoolRing.
__global__ void k_recent_sync(const float* kc, const float* vc, float* pk, float* pv, int PS, int P, int ctx,
                              const int32_t* pos0, int T) {
    const int n = min(T, kPoolRing * kSelBS);
    const int pos = *pos0 + T - n + blockIdx.x, kh = blockIdx.y;
    const size_t src = ((size_t) kh * ctx + pos) * kSelHD + threadIdx.x;
    const size_t dst = (((size_t) kh * PS + P + ((pos / kSelBS) & (kPoolRing - 1))) * kSelBS + (pos & (kSelBS - 1))) * kSelHD +
                       threadIdx.x;
    pk[dst] = kc[src];
    pv[dst] = vc[src];
}

/// A prompt chunk / verify window starting at pos0 rewrites the cells from there on: forget the cached blocks from there.
/// grid HKV, block 256.
__global__ void k_pool_inval(int32_t* tags, int P, const int32_t* pos0) {
    const int first = *pos0 / kSelBS;
    int32_t* tg = tags + (size_t) blockIdx.x * P;
    for (int s = threadIdx.x; s < P; s += blockDim.x)
        if (tg[s] >= first) tg[s] = -1;
}

/// One decode token: map the 96 selected blocks of every KV head to pool slots (hit: the slot; miss: the least recently used
/// slot not used in this step; the newest blocks: direct).  grid HKV, block 256, P <= 256, P >= kSelN.
__global__ void k_pool_map(const int32_t* ids, int32_t* slots, int32_t* tags, int32_t* last, int32_t* state, int32_t* stat,
                           int P, const int32_t* pos0) {
    __shared__ int sel_slot[kSelN], miss_list[kSelN], victim[kSelN];
    __shared__ int stg[512], slu[512];   // the tags and last-use stamps of this KV head's slots, scanned from shared memory
    __shared__ int nmiss, stamp, nlook;
    const int kh = blockIdx.x, tid = threadIdx.x;
    int32_t* tg = tags + (size_t) kh * P;
    int32_t* lu = last + (size_t) kh * P;
    const int lastb = *pos0 / kSelBS;
    if (tid == 0) {
        stamp = ++state[kh];
        nmiss = 0;
        nlook = 0;
    }
    if (tid < P) {
        stg[tid] = tg[tid];
        slu[tid] = lu[tid];
    }
    __syncthreads();
    if (tid < kSelN) {
        const int id = ids[kh * kSelN + tid];
        int v = -1;
        if (id >= 0 && id < lastb - kPoolDirect) {
            for (int s = 0; s < P; ++s)
                if (stg[s] == id) {
                    v = s;
                    break;
                }
            atomicAdd(&nlook, 1);
            if (v >= 0) slu[v] = stamp;
            else {
                v = -2;
                miss_list[atomicAdd(&nmiss, 1)] = tid;
            }
        }
        else if (id >= 0) v = P + (id & (kPoolRing - 1));   // a newest block: its ring slot
        sel_slot[tid] = v;
    }
    __syncthreads();
    if (tid < P) {   // the rank of this slot among the slots not used in this step, oldest first
        const int mine = slu[tid];
        if (mine != stamp) {
            int rank = 0;
            for (int t = 0; t < P; ++t) {
                const int o = slu[t];
                if (o != stamp && (o < mine || (o == mine && t < tid))) ++rank;
            }
            if (rank < nmiss) victim[rank] = tid;
        }
    }
    __syncthreads();
    if (tid < nmiss) {
        const int i = miss_list[tid], s = victim[tid];
        stg[s] = ids[kh * kSelN + i];
        slu[s] = stamp;
        sel_slot[i] = s | kPoolMiss;
    }
    __syncthreads();
    if (tid < P) {
        tg[tid] = stg[tid];
        lu[tid] = slu[tid];
    }
    if (tid < kSelN) slots[kh * kSelN + tid] = sel_slot[tid];
    if (tid == 0) {
        atomicAdd(&stat[0], nmiss);
        atomicAdd(&stat[1], nlook);
    }
}

/// Copies the missed blocks from host memory into their pool slots.  grid (kSelN, HKV), block 256.
__global__ void k_pool_copy(const float* kc, const float* vc, const int32_t* ids, const int32_t* slots, float* pk, float* pv,
                            int P, int ctx) {
    const int i = blockIdx.x, kh = blockIdx.y;
    const int v = slots[kh * kSelN + i];
    if (v < 0 || !(v & kPoolMiss)) return;
    const int s = v & 0xffff, id = ids[kh * kSelN + i];
    const size_t n4 = (size_t) kSelBS * kSelHD / 4;
    const float4* sk = reinterpret_cast<const float4*>(kc + ((size_t) kh * ctx + (size_t) id * kSelBS) * kSelHD);
    const float4* sv = reinterpret_cast<const float4*>(vc + ((size_t) kh * ctx + (size_t) id * kSelBS) * kSelHD);
    float4* dk = reinterpret_cast<float4*>(pk + ((size_t) kh * P + s) * kSelBS * kSelHD);
    float4* dv = reinterpret_cast<float4*>(pv + ((size_t) kh * P + s) * kSelBS * kSelHD);
    for (size_t j = threadIdx.x; j < n4; j += blockDim.x) {
        dk[j] = sk[j];
        dv[j] = sv[j];
    }
}

__device__ __forceinline__ float sel_wmax(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, 32));
    return x;
}

/// Split-K attention over the selected blocks, one query token: one block per (chunk of 4 selected blocks, KV head), all 8
/// query heads of the KV head together so every K / V row is read once (a cell is in the pool, or in host memory when its
/// slot is -1 / slots is null: the rows are read whole, one warp per cell).  grid (kSelN / 4, HKV), block 256, HD == 256.
/// Partials as k_attn_part's (chunk c of query head h = kvh * 8 + g at h * n_chunk + c).
__global__ void k_sel_part(const float* q, const float* kc, const float* vc, const int32_t* ids, const int32_t* slots,
                           const float* pk, const float* pv, int P, const int32_t* pos0, int ctx, float scale, int n_chunk,
                           float* pacc, float* pm, float* pl) {
    __shared__ float qs[kSelG * kSelHD], sc[kSelG * kChunkCells];
    __shared__ int bk[4], bs[4];
    const int c = blockIdx.x, kvh = blockIdx.y;
    const int npos = *pos0 + 1;
    const int j = threadIdx.x, warp = j >> 5, lane = j & 31;
    if (j < 4) {
        bk[j] = ids[kvh * kSelN + c * 4 + j];
        const int v = slots ? slots[kvh * kSelN + c * 4 + j] : -1;
        bs[j] = v < 0 ? -1 : (v & 0xffff);
    }
    for (int i = j; i < kSelG * kSelHD; i += blockDim.x) qs[i] = q[(size_t) kvh * kSelG * kSelHD + i];
    __syncthreads();
    for (int k = warp; k < kChunkCells; k += 8) {   // scores of the 8 query heads for cell k
        const int blk = bk[k >> 6];
        const int p = blk * kSelBS + (k & 63);
        if (blk >= 0 && p < npos) {
            const int sl = bs[k >> 6];
            const float* kr = sl >= 0 ? pk + (((size_t) kvh * P + sl) * kSelBS + (k & 63)) * kSelHD
                                      : kc + ((size_t) kvh * ctx + p) * kSelHD;
            const float4* kp = reinterpret_cast<const float4*>(kr);
            const float4 a = kp[lane], b = kp[lane + 32];
            for (int g = 0; g < kSelG; ++g) {
                const float* qg = qs + g * kSelHD;
                float s = qg[lane * 4] * a.x + qg[lane * 4 + 1] * a.y + qg[lane * 4 + 2] * a.z + qg[lane * 4 + 3] * a.w +
                          qg[128 + lane * 4] * b.x + qg[128 + lane * 4 + 1] * b.y + qg[128 + lane * 4 + 2] * b.z +
                          qg[128 + lane * 4 + 3] * b.w;
                s = sel_wsum(s);
                if (lane == 0) sc[g * kChunkCells + k] = s * scale;
            }
        } else if (lane == 0) {
            for (int g = 0; g < kSelG; ++g) sc[g * kChunkCells + k] = -INFINITY;
        }
    }
    __syncthreads();
    {   // warp g: the chunk's max and sum for query head g, the scores turned into weights
        float m = -INFINITY;
        for (int k = lane; k < kChunkCells; k += 32) m = fmaxf(m, sc[warp * kChunkCells + k]);
        m = sel_wmax(m);
        float l = 0.f;
        for (int k = lane; k < kChunkCells; k += 32) {
            const float s = sc[warp * kChunkCells + k];
            const float e = s == -INFINITY ? 0.f : expf(s - m);
            sc[warp * kChunkCells + k] = e;
            l += e;
        }
        l = sel_wsum(l);
        if (lane == 0) {
            const size_t o = (size_t) (kvh * kSelG + warp) * n_chunk + c;
            pm[o] = m;
            pl[o] = l;
        }
    }
    __syncthreads();
    float acc[kSelG];
    for (int g = 0; g < kSelG; ++g) acc[g] = 0.f;
    for (int k = 0; k < kChunkCells; ++k) {
        const int blk = bk[k >> 6];
        const int cell = blk * kSelBS + (k & 63);
        if (blk < 0 || cell >= npos) continue;
        const int sl = bs[k >> 6];
        const float v = sl >= 0 ? pv[(((size_t) kvh * P + sl) * kSelBS + (k & 63)) * kSelHD + j]
                                : vc[((size_t) kvh * ctx + cell) * kSelHD + j];
        for (int g = 0; g < kSelG; ++g) acc[g] += sc[g * kChunkCells + k] * v;
    }
    for (int g = 0; g < kSelG; ++g) pacc[((size_t) (kvh * kSelG + g) * n_chunk + c) * kSelHD + j] = acc[g];
}

/// The chunks merged, times sigmoid(gate): grid H, block HD (one token).
__global__ void k_sel_merge(const float* pacc, const float* pm, const float* pl, const float* gate, int n_chunk, float* out) {
    const int row = blockIdx.x, d = threadIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < kSelN / 4; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < kSelN / 4; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * kSelHD + d];
        den += w * pl[o];
    }
    const size_t gi = (size_t) row * kSelHD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

