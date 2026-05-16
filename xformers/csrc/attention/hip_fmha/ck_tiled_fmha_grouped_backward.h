/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/pinned_host_releaser.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/fmha.hpp>

#include "ck_fmha_util.h"
#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_bwd_setting.h"
#include "ck_tiled_fmha_params.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck_tile::index_t MaxK>
struct grouped_backward_mask_bias_dropout_dispatch {
  using FmhaBlockDropout =
      typename FmhaBwdBlockDropoutMaker<kHasDropout, MaxK>::dropout;

  using FmhaShape = typename FmhaBwdShape<MaxK>::Type;

  template <typename FmhaTraits, typename FmhaMask>
  using FmhaBwdPipelineProblemTemp = ck_tile::BlockFmhaBwdPipelineProblem<
      typename FmhaBwdTypeConfig<ScalarType>::QDataType,
      typename FmhaBwdTypeConfig<ScalarType>::KDataType,
      typename FmhaBwdTypeConfig<ScalarType>::VDataType,
      typename FmhaBwdTypeConfig<ScalarType>::GemmDataType,
      typename FmhaBwdTypeConfig<ScalarType>::LSEDataType,
      typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
      typename FmhaBwdTypeConfig<ScalarType>::DDataType,
      typename FmhaBwdTypeConfig<ScalarType>::BiasDataType,
      typename FmhaBwdTypeConfig<ScalarType>::RandValOutputDataType,
      typename FmhaBwdTypeConfig<ScalarType>::ODataType,
      typename FmhaBwdTypeConfig<ScalarType>::OGradDataType,
      typename FmhaBwdTypeConfig<ScalarType>::QGradDataType,
      typename FmhaBwdTypeConfig<ScalarType>::KGradDataType,
      typename FmhaBwdTypeConfig<ScalarType>::VGradDataType,
      typename FmhaBwdTypeConfig<ScalarType>::BiasGradDataType,
      FmhaShape,
      true, // kIsGroupMode
      false, // non-deterministic
      FmhaMask,
      FmhaBlockDropout,
      false, // kUseTrLoad, not used
      FmhaTraits>;

  static void Run(GroupedBackwardParams& param, hipStream_t stream) {
    {
      constexpr ck_tile::index_t kBlockSize = 64;
      bool pad_headdim_v = !(param.Kv % MaxK == 0);

      constexpr bool kPadSeqLenQ = true;

      BOOL_SWITCH(pad_headdim_v, kPadHeadDimV, [&] {
        constexpr ck_tile::index_t occupancy = 2;

        using FmhaOGradDotOTraits_ = ck_tile::
            TileFmhaBwdOGradDotOTraits<kPadSeqLenQ, kPadHeadDimV, occupancy>;

        using FmhaBwdOGradDotOPipelineProblem =
            ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
                typename FmhaBwdTypeConfig<ScalarType>::ODataType,
                typename FmhaBwdTypeConfig<ScalarType>::OGradDataType,
                typename FmhaBwdTypeConfig<ScalarType>::DDataType,
                typename FmhaBwdTypeConfig<ScalarType>::LSEDataType,
                kBlockSize,
                MaxK, // kVHeaddim
                true, // kIsGroupMode
                FmhaOGradDotOTraits_>;

        using FmhaBwdOGradDotOPipeline_ =
            typename ck_tile::BlockFmhaBwdOGradDotO<
                FmhaBwdOGradDotOPipelineProblem>;

        using FmhaBwdOGradDotOKernel_ =
            ck_tile::FmhaBwdOGradDotOKernel<FmhaBwdOGradDotOPipeline_>;

        RunWithBwdOGradDotOKernel<FmhaBwdOGradDotOKernel_>(param, stream);
      });
    };

    {
      constexpr ck_tile::index_t occupancy = 1;
      const bool has_dropout = (param.dropout_prob > 0.0f);

      using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<kHasMask>;

      constexpr auto kBiasEnum = kHasBias
          ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
          : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

      const bool pad_headdim_q = !(param.K % FmhaShape::kQKHeaddim == 0);
      const bool pad_headdim_v = !(param.Kv % FmhaShape::kVHeaddim == 0);

      BOOL_SWITCH_2(
          pad_headdim_q, kPadHeadDimQ, pad_headdim_v, kPadHeadDimV, [&] {
            using FmhaBwdTraits_ = ck_tile::TileFmhaBwdTraits<
                kPadHeadDimQ,
                kPadHeadDimV,
                kBiasEnum,
                kHasBiasGrad,
                occupancy>;

            using FmhaBwdPipelineProblem =
                FmhaBwdPipelineProblemTemp<FmhaBwdTraits_, FmhaMask>;

            using FmhaBwdPipeline_ =
                typename ck_tile::BlockFmhaBwdDQDKDVPipelineSelector<
                    FmhaBwdPipelineProblem,
                    void>::type;

            using FmhaBwdKGradEpilogue_ =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
                    typename FmhaBwdTypeConfig<ScalarType>::KGradDataType,
                    false, // kPadSeqLenK,
                    kPadHeadDimQ>>;

            using FmhaBwdVGradEpilogue_ =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
                    typename FmhaBwdTypeConfig<ScalarType>::VGradDataType,
                    false, // kPadSeqLenK,
                    kPadHeadDimV>>;

            using FmhaBwdDQDKDVKernel_ = ck_tile::FmhaBwdDQDKDVKernel<
                FmhaBwdPipeline_,
                FmhaBwdKGradEpilogue_,
                FmhaBwdVGradEpilogue_>;

            RunWithBwdDQDKDVKernel<FmhaBwdDQDKDVKernel_>(param, stream);
          });
    };

    {
      constexpr ck_tile::index_t kBlockSize = 128;

      const bool pad_seqlen_q = true;
      const bool pad_headdim_q = !(param.K % MaxK == 0);

      BOOL_SWITCH_2(
          pad_seqlen_q, kPadSeqLenQ, pad_headdim_q, kPadHeadDimQ, [&] {
            constexpr ck_tile::index_t occupancy = 2;

            using FmhaBwdConvertQGradTraits_ =
                ck_tile::TileFmhaBwdConvertQGradTraits<
                    kPadSeqLenQ,
                    kPadHeadDimQ,
                    occupancy>;

            using FmhaBwdConvertQGradPipelineProblem =
                ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<
                    typename FmhaBwdTypeConfig<ScalarType>::AccDataType,
                    typename FmhaBwdTypeConfig<ScalarType>::QGradDataType,
                    kBlockSize,
                    64, // kM0
                    MaxK, // kQKHeaddim
                    true, // kIsGroupMode
                    false, // kIsDeterministic
                    FmhaBwdConvertQGradTraits_>;

            using FmhaBwdConvertQGradPipeline =
                typename ck_tile::BlockFmhaBwdConvertQGrad<
                    FmhaBwdConvertQGradPipelineProblem>;

            using FmhaBwdConvertQGradKernel_ =
                ck_tile::FmhaBwdConvertQGradKernel<FmhaBwdConvertQGradPipeline>;

            RunWithBwdConvertQGradKernel<FmhaBwdConvertQGradKernel_>(
                param, stream);
          });
    };
  }

  template <typename FmhaBwdOGradDotOKernel>
  static void RunWithBwdOGradDotOKernel(
      GroupedBackwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaBwdOGradDotOKernel::MakeKargs(
          param.out_ptr,
          param.grad_out_ptr,
          param.dot_out_ptr,
          nullptr, // lse_ptr
          nullptr, // sink_ptr
          nullptr, // d_sink_ptr
          1.0f - param.dropout_prob,
          param.seqstart_q_dev_ptr,
          nullptr, // seqlen_q_ptr, most recently added kernel argument
          nullptr, // cu_seqlen_q_ptr, most recently added kernel argument
          param.Kv,
          param.Hq, // nhead_q
          param.grad_out_strides[0], // stride_do
          param.out_strides[0], // stride_o
          param.grad_out_strides[1], // nhead_stride_do
          param.out_strides[1], // nhead_stride_o
          param.lsed_strides[0]); // nhead_stride_d
    }();

    dim3 kGridSize = FmhaBwdOGradDotOKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q);
    dim3 kBlockSize = FmhaBwdOGradDotOKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu =
        FmhaBwdOGradDotOKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            FmhaBwdOGradDotOKernel{}, kGridSize, kBlockSize, 0, kargs));
  }

  template <typename FmhaBwdDQDKDVKernel>
  static void RunWithBwdDQDKDVKernel(
      GroupedBackwardParams& param,
      hipStream_t stream) {
    if (param.seqstart_q_host_ptr != nullptr) {
      auto host_ws_size =
          FmhaBwdDQDKDVKernel::GetWorkspaceHostSize(param.num_batches);

      void* host_buf;

      HIP_CALL_CHECK(hipHostMalloc(&host_buf, host_ws_size));
      auto device_ws_size = FmhaBwdDQDKDVKernel::PrepareWorkspaceHost(
          host_buf,
          param.num_batches,
          param.K,
          param.Hq,
          param.M,
          param.N,
          param.seqstart_q_host_ptr,
          param.seqstart_k_host_ptr);
      param.workspace_size = host_ws_size + device_ws_size;

      char* gpu_buf;

      HIP_CALL_CHECK(hipMallocAsync(&gpu_buf, param.workspace_size, stream));
      HIP_CALL_CHECK(hipMemcpyAsync(
          gpu_buf, host_buf, host_ws_size, hipMemcpyHostToDevice, stream));

      // to release the host buffer in async
      HIP_CALL_CHECK(hipLaunchHostFunc(
          stream,
          [](void* ud) {
            ck_tile::pinned_host_releaser::instance().enqueue(ud);
          },
          host_buf));

      if (FmhaBwdDQDKDVKernel::NeedsZeroDqAcc()) {
        HIP_CALL_CHECK(
            hipMemsetAsync(gpu_buf + host_ws_size, 0, device_ws_size, stream));
      };

      param.workspace_ptr = reinterpret_cast<uint8_t*>(gpu_buf);
    } else {
      auto host_ws_size =
          FmhaBwdDQDKDVKernel::GetWorkspaceHostSize(param.num_batches);

      const size_t seqstart_bytes = sizeof(int) * (param.num_batches + 1);
      const size_t pin_total = 2 * seqstart_bytes + host_ws_size;
      char* pin_buf;

      HIP_CALL_CHECK(hipHostMalloc(&pin_buf, pin_total));

      int* pin_q = reinterpret_cast<int*>(pin_buf);
      int* pin_k = reinterpret_cast<int*>(pin_buf + seqstart_bytes);
      void* pin_w = pin_buf + 2 * seqstart_bytes;

      // to prepare the seqstart_q_host[] and seqstart_k_host[] in async
      HIP_CALL_CHECK(hipMemcpyAsync(
          pin_q,
          param.seqstart_q_dev_ptr,
          seqstart_bytes,
          hipMemcpyDeviceToHost,
          stream));
      HIP_CALL_CHECK(hipMemcpyAsync(
          pin_k,
          param.seqstart_k_dev_ptr,
          seqstart_bytes,
          hipMemcpyDeviceToHost,
          stream));

      struct PrepareCtx {
        void* pin_w;
        int* pin_q;
        int* pin_k;
        int batch;
        int hdim_q;
        int nhead_q;
      };
      auto* ctx = new PrepareCtx{
          pin_w, pin_q, pin_k, param.num_batches, param.K, param.Hq};

      // to construct the content in host workspace in async
      HIP_CALL_CHECK(hipLaunchHostFunc(
          stream,
          [](void* ud) {
            auto* c = static_cast<PrepareCtx*>(ud);
            FmhaBwdDQDKDVKernel::PrepareWorkspaceHost(
                c->pin_w,
                c->batch,
                c->hdim_q,
                c->nhead_q,
                0, // seqlen_q, unused in group mode
                0, // seqlen_k unused in group mode
                c->pin_q,
                c->pin_k);
            delete c;
          },
          ctx));

      const size_t device_ws_size =
          FmhaBwdDQDKDVKernel::GetWorkspaceDeviceSizeUpperBound(
              param.num_batches,
              param.K,
              param.Hq,
              param.M,
              param.max_seqlen_k);
      param.workspace_size = host_ws_size + device_ws_size;

      char* gpu_buf;

      HIP_CALL_CHECK(hipMallocAsync(&gpu_buf, param.workspace_size, stream));

      // to transfer the content of host workspace to gpu in async
      HIP_CALL_CHECK(hipMemcpyAsync(
          gpu_buf, pin_w, host_ws_size, hipMemcpyHostToDevice, stream));

      // to release the host buffer (include the seqstart_q/k_host[] and host
      // workspace) in async
      HIP_CALL_CHECK(hipLaunchHostFunc(
          stream,
          [](void* ud) {
            ck_tile::pinned_host_releaser::instance().enqueue(ud);
          },
          pin_buf));

      if (FmhaBwdDQDKDVKernel::NeedsZeroDqAcc())
        HIP_CALL_CHECK(
            hipMemsetAsync(gpu_buf + host_ws_size, 0, device_ws_size, stream));

      param.workspace_ptr = reinterpret_cast<uint8_t*>(gpu_buf);
    };

    const auto kargs = [&] {
      return FmhaBwdDQDKDVKernel::MakeKargsImpl(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.attn_bias_ptr,
          param.logsumexp_ptr,
          param.grad_out_ptr,
          param.dot_out_ptr,
          nullptr, // randval_ptr
          nullptr, // dq_ptr, only used for QrQtrDor pipeline
          param.grad_k_ptr,
          param.grad_v_ptr,
          param.grad_bias_ptr,
          param.workspace_ptr,
          param.seqstart_q_dev_ptr,
          param.seqstart_k_dev_ptr,
          nullptr, // seqlen_q_ptr, most recently added kernel argument
          param.seqlen_k_dev_ptr,
          nullptr, // cu_seqlen_q_ptr, most recently added kernel argument
          nullptr, // cu_seqlen_k_ptr, most recently added kernel argument
          0, // batch
          param.K,
          param.Kv,
          param.Hq,
          param.Hq / param.Hkv,
          param.scale,
          param.q_strides[0], // q, k, v, bias, do, dq_f32, dk, dv, dbias
                              // seq-dim stride
          param.k_strides[0],
          param.v_strides[0],
          param.attn_bias_strides[1],
          0, // stride_randval
          param.grad_out_strides[0],
          0, // stride_dq, // only used for QrQtrDor pipeline
          param.grad_k_strides[0],
          param.grad_v_strides[0],
          param.attn_bias_strides[1], // assume grad_bias has same strides as
                                      // bias.
          param.q_strides[1], // q, k, v, bias, do, lse/dot, dq_f32, dk, dv,
                              // dbias nhead-dim strides
          param.k_strides[1],
          param.v_strides[1],
          param.attn_bias_strides[0],
          0, // nhead_stride_randval
          param.grad_out_strides[1],
          param.lsed_strides[0], // assume lse/dot is in HM contiguous layout
          0, // nhead_stride_dq, // only used for QrQtrDor pipeline
          param.grad_k_strides[1],
          param.grad_v_strides[1],
          param.attn_bias_strides[0], // assume grad_bias has same strides as
                                      // bias
          (param.window_size > 0) ? param.window_size - 1
                                  : -1, // window_left_size
          (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
          param.custom_mask_type,
          param.dropout_prob, // dropout ratio
          std::make_pair(param.philox_seed, param.philox_offset));
    }();

    dim3 kGridSize = FmhaBwdDQDKDVKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_k);
    dim3 kBlockSize = FmhaBwdDQDKDVKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaBwdDQDKDVKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            FmhaBwdDQDKDVKernel{}, kGridSize, kBlockSize, 0, kargs));
  }

  template <typename FmhaBwdConvertQGradKernel>
  static void RunWithBwdConvertQGradKernel(
      GroupedBackwardParams& param,
      hipStream_t stream) {
    const auto kargs = [&] {
      return FmhaBwdConvertQGradKernel::MakeKargs(
          param.workspace_ptr,
          param.grad_q_ptr,
          param.num_batches,
          param.Hq,
          param.seqstart_q_dev_ptr,
          param.seqstart_k_dev_ptr,
          nullptr, // seqlen_q_ptr, most recently added kernel argument
          param.seqlen_k_dev_ptr, // most recently used kernel argument
          nullptr, // cu_seqlen_q_ptr, most recently added kernel argument
          nullptr, // cu_seqlen_k_ptr, most recently added kernel argument
          param.K, // headdim of q/k
          param.q_strides[0],
          param.q_strides[1]);
    }();

    dim3 kGridSize = FmhaBwdConvertQGradKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q);
    dim3 kBlockSize = FmhaBwdConvertQGradKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu =
        FmhaBwdConvertQGradKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            FmhaBwdConvertQGradKernel{}, kGridSize, kBlockSize, 0, kargs));

    if (param.workspace_size > 0) {
      HIP_CALL_CHECK(hipFreeAsync(param.workspace_ptr, stream));
    };
  }
};

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasBiasGrad,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_grouped_backward_mask_bias_dropout_dispatch(
    GroupedBackwardParams& param,
    hipStream_t stream) {
  grouped_backward_mask_bias_dropout_dispatch<
      ScalarType,
      kHasMask,
      kHasBias,
      kHasBiasGrad,
      kHasDropout,
      MaxK>::Run(param, stream);
};
