/*
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>

#include "ck_tile_attention_inner_product.h"

namespace {
template <typename data_t, int32_t vec_size>
__device__ ck_tile::ext_vector_t<float, vec_size> scalar_scale_acc(
    ck_tile::ext_vector_t<float, vec_size> acc,
    ck_tile::ext_vector_t<data_t, vec_size> a,
    float b) {
  union {
    decltype(acc) vec;
    float arr[vec_size];
  } acc_u{acc};
  union {
    decltype(a) vec;
    data_t arr[vec_size];
  } a_u{a};

#pragma unroll
  for (int32_t i = 0; i < vec_size; ++i) {
    acc_u.arr[i] += ck_tile::type_convert<float>(a_u.arr[i]) * b;
  }

  return acc_u.vec;
}

template <typename F, int32_t n_threads_per_wavefront = 64>
float __device__ __forceinline__ wavefrontReduce(float val, F f) {
#pragma unroll
  for (int32_t mask = n_threads_per_wavefront >> 1; mask > 0; mask >>= 1) {
    val = f(__shfl_xor(val, mask, n_threads_per_wavefront), val);
  }
  return val;
}

template <typename TData, typename TDataVec>
__forceinline__ __device__ void load_v(
    const TData* __restrict__ data_ptr,
    int32_t vector_offset,
    TDataVec* __restrict__ load_to) {
  *load_to = *(reinterpret_cast<const TDataVec*>(data_ptr) + vector_offset);
}

template <typename TData, typename TDataVec>
__forceinline__ __device__ void store_v(
    TData* __restrict__ data_ptr,
    int32_t vector_offset,
    TDataVec value) {
  *(reinterpret_cast<TDataVec*>(data_ptr) + vector_offset) = value;
}
} // namespace

namespace ck_tile {

template <typename scalar_t>
struct ForwardDecoderAttnKernelArg {
  const scalar_t* __restrict__ XQ;
  const scalar_t* __restrict__ cache_K;
  const scalar_t* __restrict__ cache_V;
  scalar_t* __restrict__ O;
  const int32_t* __restrict__ seq_kv_lens;
  const ptrdiff_t XQ_stride_b;
  const ptrdiff_t XQ_stride_m;
  const ptrdiff_t XQ_stride_g;
  const ptrdiff_t XQ_stride_h;
  const ptrdiff_t K_stride_b;
  const ptrdiff_t K_stride_m;
  const ptrdiff_t K_stride_g;
  const ptrdiff_t K_stride_h;
  const int32_t Q_size_m;
  const int32_t Q_size_g;
  const int32_t Q_size_h;
  const int32_t Q_size_k;
  const int32_t K_size_m;
  const bool multiquery;
  const float qk_scale;
};

template <
    typename scalar_t,
    int32_t vec_size = 4,
    int32_t n_loop_unroll = 16,
    int32_t n_loop_unroll_tail = 2,
    int32_t KV_M_MAX = 8192,
    int32_t n_wavefronts_per_block = 16>
struct ForwardDecoderAttnKernelImpl {
// clang-format off
CK_TILE_DEVICE void operator()(ForwardDecoderAttnKernelArg<scalar_t> arg) {
  static_assert(n_loop_unroll_tail < n_loop_unroll, "");

  // Each block handles a single batch and head and query and group
  const int32_t b = blockIdx.x / (arg.Q_size_m * arg.Q_size_g * arg.Q_size_h);
  const int32_t m =
      (blockIdx.x / (arg.Q_size_g * arg.Q_size_h)) % arg.Q_size_m;
  const int32_t g = (blockIdx.x / arg.Q_size_h) % arg.Q_size_g;
  const int32_t h = blockIdx.x % arg.Q_size_h;

  // Note: this is decoding case where we attend to current and all previous
  // tokens.
  const int32_t t_max = arg.seq_kv_lens ? arg.seq_kv_lens[b] : arg.K_size_m;

  const int32_t lane_idx = threadIdx.x;
  const int32_t wavefront_idx = threadIdx.y;
  const int32_t threads_per_wavefront = blockDim.x;
  const int32_t wavefronts_per_block = blockDim.y;
  const int32_t threads_per_block =
      threads_per_wavefront * wavefronts_per_block;
  const int32_t thread_linear_idx =
      lane_idx + wavefront_idx * threads_per_wavefront;
  // const auto* q_ = &(XQ_acc[b][m][g][h][0]);
  const auto XQO_base_offset = b * arg.XQ_stride_b + m * arg.XQ_stride_m +
      g * arg.XQ_stride_g + h * arg.XQ_stride_h;
  const auto* __restrict__ q_ = arg.XQ + XQO_base_offset;

  const auto cache_KV_base_offset = b * arg.K_stride_b + 0 * arg.K_stride_m +
      g * arg.K_stride_g + (arg.multiquery ? 0 : h * arg.K_stride_h);
  const auto* __restrict__ cache_K_base = arg.cache_K + cache_KV_base_offset;
  const auto* __restrict__ cache_V_base = arg.cache_V + cache_KV_base_offset;

  using data_t = scalar_t;
  using compute_t = float;
  using data_vec_t = std::conditional_t<
      vec_size == 1,
      data_t,
      ck_tile::ext_vector_t<data_t, vec_size>>;
  using compute_vec_t = ck_tile::ext_vector_t<compute_t, vec_size>;

  const bool lane_active_for_io = lane_idx * vec_size < arg.Q_size_k;

  extern __shared__ __align__(16) compute_t smem[];

  data_vec_t q_thread = 0;
  // Load Q into registers in all wavefronts.
  // Each thread handles `vec_size` D dimensions
  if (lane_active_for_io) {
    load_v<data_t, data_vec_t>(q_, lane_idx, &q_thread);
  }

  compute_t max_qk_acc = ck_tile::numeric<compute_t>::lowest();

  // Compute S[0:t_max] =
  // ```
  // for t in range(t_max):
  //   S[t] = dot(Q, K[t])
  // ```
  // Split the 0:t_max range across wavefronts in a block,
  // unroll loads to expose more parallelism.
  // Reduce the dot product with cross-lane operation;
  // Q and K[t] are in the registers of threads in a single wavefront.

  data_vec_t k_loads[n_loop_unroll] = {};

  constexpr auto dtt = n_wavefronts_per_block * n_loop_unroll;
  const int32_t t_max_unroll = (t_max / dtt) * dtt;

  for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll;
        tt += dtt) {
    if (lane_active_for_io) {
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        const int32_t t = tt + ttt;
        // load the K[b][t][g][h|0][:] row into registers
        load_v<data_t, data_vec_t>(
            cache_K_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
      }
    }
    compute_t qk_accs[n_loop_unroll] = {};
#pragma unroll n_loop_unroll
    for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
      ck_tile::inner_product<data_vec_t, data_vec_t, compute_t>(
          q_thread, k_loads[ttt], qk_accs[ttt]);
      qk_accs[ttt] *= arg.qk_scale;

      qk_accs[ttt] =
          wavefrontReduce(qk_accs[ttt], [](auto a, auto b) { return a + b; });
      max_qk_acc = ck_tile::max(qk_accs[ttt], max_qk_acc);
    }
    if (lane_idx == 0) {
      auto* __restrict__ smem_base = smem + tt;
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        smem_base[ttt] = qk_accs[ttt];
      }
    }
  }

  // NB: the length of the tail is <= (wavefronts_per_block * n_loop_unroll)
  for (auto tt = t_max_unroll + wavefront_idx * n_loop_unroll_tail;
        tt < t_max;
        tt += wavefronts_per_block * n_loop_unroll_tail) {
    if (lane_active_for_io) {
#pragma unroll n_loop_unroll_tail
      for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
        const int32_t t = tt + ttt;
        if (t < t_max) {
          // load the K[b][t][g][h|0][:] row into registers
          load_v<data_t, data_vec_t>(
              cache_K_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
        }
      }
    }
#pragma unroll n_loop_unroll_tail
    for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
      compute_t qk_acc = 0;
      const int32_t t = tt + ttt;
      if (t < t_max) {
        ck_tile::inner_product<data_vec_t, data_vec_t, compute_t>(
            q_thread, k_loads[ttt], qk_acc);
        qk_acc *= arg.qk_scale;

        qk_acc =
            wavefrontReduce(qk_acc, [](auto a, auto b) { return a + b; });
        max_qk_acc = ck_tile::max(qk_acc, max_qk_acc);

        // write accumulated sums to smem.
        if (lane_idx == 0) {
          smem[t] = qk_acc;
        }
      }
    }
  }

  // Use shared reduction to compute max and compute softmax on shared memory.
  // write max acc
  if (lane_idx == 0) {
    smem[KV_M_MAX + wavefront_idx] = max_qk_acc;
  }
  __syncthreads();
  if (lane_idx < wavefronts_per_block) {
    max_qk_acc = ck_tile::max(max_qk_acc, smem[KV_M_MAX + lane_idx]);
  }
  // shared across all threads in block
  max_qk_acc = wavefrontReduce(
      max_qk_acc, [](auto a, auto b) { return a > b ? a : b; });

  // each wavefront computes partial sum of exp.
  compute_t softmax_denominator = 0.0f;
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    softmax_denominator += ck_tile::exp(smem[t] - max_qk_acc);
  }
  softmax_denominator = wavefrontReduce(
      softmax_denominator, [](auto a, auto b) { return a + b; });

  if (lane_idx == 0) {
    smem[KV_M_MAX + wavefront_idx] = softmax_denominator;
  }
  __syncthreads();

  // now, compute sum of exp(x - max(x)) over all intermediate results.
  softmax_denominator = 0.0;
  if (lane_idx < wavefronts_per_block) {
    softmax_denominator = smem[KV_M_MAX + lane_idx];
  }
  softmax_denominator = wavefrontReduce(
      softmax_denominator, [](auto a, auto b) { return a + b; });

  const compute_t softmax_scale_factor = 1. / softmax_denominator;
  // now, compute the normalization across all threads.
  for (int32_t t = thread_linear_idx; t < t_max; t += threads_per_block) {
    smem[t] = ck_tile::exp(smem[t] - max_qk_acc) * softmax_scale_factor;
  }
  __syncthreads();

  // Split T across wavefronts in a block
  // each wavefront compute sum(t_subset) P[t] * V[t_subset, d]
  // outputs are of size float[D]

  compute_t ps[n_loop_unroll] = {};
  compute_vec_t o_acc = 0;
  if (lane_active_for_io) {
    for (auto tt = wavefront_idx * n_loop_unroll; tt < t_max_unroll;
          tt += dtt) {
#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        const int32_t t = tt + ttt;
        // load the V[b][t][g][h|0][:] row into registers, reusing K register
        // storage
        load_v<data_t, data_vec_t>(
            cache_V_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
        ps[ttt] = smem[t];
      }

#pragma unroll n_loop_unroll
      for (auto ttt = 0; ttt < n_loop_unroll; ++ttt) {
        o_acc =
            scalar_scale_acc<data_t, vec_size>(o_acc, k_loads[ttt], ps[ttt]);
      }
    }

    for (auto tt = t_max_unroll + wavefront_idx * n_loop_unroll_tail;
          tt < t_max;
          tt += wavefronts_per_block * n_loop_unroll_tail) {
#pragma unroll n_loop_unroll_tail
      for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
        const int32_t t = tt + ttt;
        if (t < t_max) {
          // load the V[b][t][g][h|0][:] row into registers, reusing K
          // register storage
          load_v<data_t, data_vec_t>(
              cache_V_base + t * arg.K_stride_m, lane_idx, &k_loads[ttt]);
          ps[ttt] = smem[t];
        }
      }

#pragma unroll n_loop_unroll_tail
      for (auto ttt = 0; ttt < n_loop_unroll_tail; ++ttt) {
        const int32_t t = tt + ttt;
        if (t < t_max) {
          o_acc = scalar_scale_acc<data_t, vec_size>(
              o_acc, k_loads[ttt], ps[ttt]);
        }
      }
    }
  }
  // now, each thread has partial sums. Write to smem and get accumulated
  // results back.
  __syncthreads();

  // NB: needs sizeof(smem) >= `vec_size` * (sizeof(float)==4) *
  // threadsPerBlock
  if (lane_active_for_io) {
    store_v<compute_t, compute_vec_t>(&smem[0], thread_linear_idx, o_acc);
  }

  __syncthreads();
  // sum up partial D rows from other wavefronts
  if (wavefront_idx == 0 && lane_active_for_io) {
    union {
      compute_vec_t vec = 0;
      compute_t arr[vec_size];
    } r;
    for (int32_t w = 0; w < wavefronts_per_block; ++w) {
      compute_vec_t partial_r;
      load_v<compute_t, compute_vec_t>(
          smem, w * threads_per_wavefront + lane_idx, &partial_r);
      r.vec += partial_r;
    }
    // elementwise convert from compute_t result to data_t out to be written
    union {
      data_vec_t vec;
      data_t arr[vec_size];
    } bf_r;
#pragma unroll
    for (int32_t i = 0; i < vec_size; ++i) {
      bf_r.arr[i] = ck_tile::type_convert<data_t>(r.arr[i]);
    }
    // write output row O[b][m][g][h][:]
    data_t* __restrict__ o_ = arg.O + XQO_base_offset;
    store_v<data_t, data_vec_t>(o_, lane_idx, bf_r.vec);
  } // write loop
} // operator()
// clang-format on
};

} // namespace ck_tile
