/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>
#include <torch/types.h>
#include <ATen/cuda/PhiloxUtils.cuh>

#include <cstdint>

#include <ck_tile/core.hpp>

namespace {

__global__ void rand_uniform_int_kernel(
    uint8_t* randvals,
    int M,
    int N,
    int num_heads,
    int64_t stride_m,
    int64_t stride_n,
    int64_t stride_head,
    int64_t stride_batch,
    uint64_t philox_seed,
    uint64_t philox_offset) {
  constexpr int kPhiloxPerTile = 64;
  constexpr int kWarpGemmMN = 32;

  const int row = blockIdx.x;
  const int col = blockIdx.y;
  const int batch_head = blockIdx.z;
  const int i_batch = batch_head / num_heads;
  const int i_head = batch_head - i_batch * num_heads;
  const int lane = threadIdx.x;

  if (lane >= kPhiloxPerTile) {
    return;
  }

  const auto subsequence =
      ck_tile::bit_cast<unsigned long long>(make_uint2(row, col));
  ck_tile::philox ph(
      philox_seed,
      philox_offset + (i_batch * num_heads + i_head) * kPhiloxPerTile + lane);

  uint8_t random_uint8_t[16];
  ph.get_random_16x8(random_uint8_t, subsequence);

  uint8_t* out = randvals +
      static_cast<int64_t>(i_batch) * stride_batch +
      static_cast<int64_t>(i_head) * stride_head;

  for (int r = 0; r < 16; ++r) {
    const int i = (16 * (r / 8) % kWarpGemmMN) + 8 * (lane / 32) + (r % 8);
    const int j = lane % kWarpGemmMN;
    const int m = row * kWarpGemmMN + i;
    const int n = col * kWarpGemmMN + j;

    if (m < M && n < N) {
      out[static_cast<int64_t>(m) * stride_m +
          static_cast<int64_t>(n) * stride_n] = random_uint8_t[r];
    }
  }
}

/**
 * generate a tensor with random uniform values. only used for testing, not much
 * attention is paid to performance
 */
at::Tensor rand_uniform_int(
    double dropout_prob,
    const at::Tensor& out_pattern) // [Batches, num_head, query_len, key_len]
{
  (void)dropout_prob;

  int B = out_pattern.size(0);
  int num_heads = out_pattern.size(1);
  int M = out_pattern.size(2);
  int N = out_pattern.size(3);

  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

  at::CUDAGeneratorImpl* gen =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(
          c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  at::PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs =
        gen->philox_cuda_state((B + 3) * (num_heads + 1) * (M + 1) * (N + 1));
  }

  const auto seeds = at::cuda::philox::unpack(rng_engine_inputs);

  int64_t philox_seed = std::get<0>(seeds);
  int64_t philox_offset = std::get<1>(seeds);

  at::Tensor randvals;

  randvals = at::empty(
      {B, num_heads, M, N}, out_pattern.options().dtype(at::ScalarType::Byte));

  if (B > 0 && num_heads > 0 && M > 0 && N > 0) {
    constexpr int kWarpGemmMN = 32;
    const dim3 grid(
        (M + kWarpGemmMN - 1) / kWarpGemmMN,
        (N + kWarpGemmMN - 1) / kWarpGemmMN,
        B * num_heads);
    const dim3 block(64);

    hipLaunchKernelGGL(
        rand_uniform_int_kernel,
        grid,
        block,
        0,
        stream,
        static_cast<uint8_t*>(randvals.data_ptr()),
        M,
        N,
        num_heads,
        randvals.stride(2),
        randvals.stride(3),
        randvals.stride(1),
        randvals.stride(0),
        static_cast<uint64_t>(philox_seed),
        static_cast<uint64_t>(philox_offset));

    const auto launch_status = hipGetLastError();
    TORCH_CHECK(
        launch_status == hipSuccess,
        "HIP rand_uniform_int_kernel launch failed: ",
        hipGetErrorString(launch_status));
  }

  const auto sync_status = hipStreamSynchronize(stream);
  TORCH_CHECK(
      sync_status == hipSuccess,
      "HIP rand_uniform_int_kernel failed: ",
      hipGetErrorString(sync_status));

  return randvals;
} // namespace

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_ck_rand_uniform"),
      TORCH_FN(rand_uniform_int));
}
