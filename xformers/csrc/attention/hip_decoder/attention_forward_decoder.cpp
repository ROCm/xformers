/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>

#include "ck_tile_attention_forward_decoder.h"

namespace {
constexpr int32_t kThreadsPerWavefront = 64;
constexpr int32_t kWavefrontsPerBlock = 16;
constexpr int32_t K_MAX = 4 * kThreadsPerWavefront;
} // namespace

namespace {

template <typename c10_t>
struct c10_to_data_t;
template <>
struct c10_to_data_t<float> {
  using type = float;
};

template <>
struct c10_to_data_t<c10::Half> {
  using type = ck_tile::fp16_t;
};

template <>
struct c10_to_data_t<c10::BFloat16> {
  using type = ck_tile::bf16_t;
};
} // namespace

namespace {

#define AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_SWITCH_3(                               \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

template <
    int32_t ThreadsPerWavefront,
    int32_t WavefrontsPerBlock,
    int32_t KV_M_MAX = 8192,
    int32_t K_MAX = K_MAX>
at::Tensor& efficient_attention_forward_decoder_ck_out_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    at::Tensor& O) {
  static_assert(4 * ThreadsPerWavefront == K_MAX, "");
  static_assert(WavefrontsPerBlock <= ThreadsPerWavefront, "");

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(!seq_kv_lens || seq_kv_lens->is_cuda());

  TORCH_CHECK(cache_K.size(1) <= KV_M_MAX);
  TORCH_CHECK(cache_K.size(4) <= K_MAX);

  constexpr auto rank = 5;

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);

  TORCH_CHECK(B <= 1024);
  TORCH_CHECK(M <= 1024);
  TORCH_CHECK(H <= 1024);

  dim3 grid_size(B * H * M * G);
  dim3 block_size(ThreadsPerWavefront, WavefrontsPerBlock);

  int32_t smem_softmax =
      KV_M_MAX * sizeof(float) + block_size.y * sizeof(float);
  int32_t smem_output = K_MAX * sizeof(float) *
      block_size
          .y; // 4 * threadsPerBlock * sizeof(float) == sizeof(O[b][0][h][:])
  const size_t lds_bytes = max(smem_softmax, smem_output);
  auto ck_stream = ck_tile::stream_config{at::hip::getCurrentHIPStream().stream()};

  AT_DISPATCH_SWITCH_3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float,
      XQ.scalar_type(),
      "efficient_attention_forward_decoder_ck",
      [&] {
        using ck_data_t = c10_to_data_t<scalar_t>::type;

        auto XQ_acc =
            XQ.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto K_acc =
            cache_K.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto V_acc =
            cache_V.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto O_acc =
            O.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto seq_acc = seq_kv_lens
            ? seq_kv_lens
                  ->packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
                  .data()
            : nullptr;
        auto arg = ck_tile::ForwardDecoderAttnKernelArg<ck_data_t>{
            reinterpret_cast<const ck_data_t* __restrict__>(XQ_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(K_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(V_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            seq_acc,
            XQ_acc.stride(0),
            XQ_acc.stride(1),
            XQ_acc.stride(2),
            XQ_acc.stride(3),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.stride(3),
            static_cast<int32_t>(XQ_acc.size(1)),
            static_cast<int32_t>(XQ_acc.size(2)),
            static_cast<int32_t>(XQ_acc.size(3)),
            static_cast<int32_t>(XQ_acc.size(4)),
            static_cast<int32_t>(K_acc.size(1)),
            K_acc.size(3) == 1,
            static_cast<float>(qk_scale)};
        auto required_vec_size = 0;

        for (auto vec_size : {4, 2, 1}) {
          if (arg.Q_size_k <= vec_size * ThreadsPerWavefront) {
            required_vec_size = vec_size;
          }
        }

        TORCH_CHECK(required_vec_size > 0);
        TORCH_CHECK(0 == arg.Q_size_k % required_vec_size);

        switch (required_vec_size) {
          case 4:
            ck_tile::launch_kernel(
                ck_stream,
                ck_tile::make_kernel(
                    ck_tile::ForwardDecoderAttnKernelImpl<ck_data_t, 4>{},
                    grid_size,
                    block_size,
                    lds_bytes,
                    arg));
            break;
          case 2:
            ck_tile::launch_kernel(
                ck_stream,
                ck_tile::make_kernel(
                    ck_tile::ForwardDecoderAttnKernelImpl<ck_data_t, 2>{},
                    grid_size,
                    block_size,
                    lds_bytes,
                    arg));
            break;
          case 1:
            ck_tile::launch_kernel(
                ck_stream,
                ck_tile::make_kernel(
                    ck_tile::ForwardDecoderAttnKernelImpl<ck_data_t, 1>{},
                    grid_size,
                    block_size,
                    lds_bytes,
                    arg));
            break;
        }
      });

  return O;
}

#undef AT_DISPATCH_CASE_3
#undef AT_DISPATCH_SWITCH_3

template <int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock>
at::Tensor efficient_attention_forward_decoder_ck_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale) {
  auto O = at::empty_like(XQ);
  efficient_attention_forward_decoder_ck_out_impl<
      ThreadsPerWavefront,
      WavefrontsPerBlock>(XQ, cache_K, cache_V, seq_kv_lens, qk_scale, O);
  return O;
}

at::Tensor efficient_attention_forward_decoder_ck(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, KV_M_MAX, G, H or 1, D]
    const at::Tensor& cache_V, // [B, KV_M_MAX, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale) {
  return efficient_attention_forward_decoder_ck_impl<
      kThreadsPerWavefront,
      kWavefrontsPerBlock>(XQ, cache_K, cache_V, seq_kv_lens, qk_scale);
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_decoder_ck"),
      TORCH_FN(efficient_attention_forward_decoder_ck));
}
