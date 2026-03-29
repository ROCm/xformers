/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/fmha.hpp>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_fwd_setting.h"
#include "ck_tiled_fmha_params.h"
#include "ck_tiled_headdim_switch.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK,
    ck_tile::index_t MTile>
struct grouped_infer_mask_bias_dropout_dispatch {
  static constexpr bool kUseWholeKPrefetchPipeline =
      (MaxK <= 128 && !kHasDropout);

#if defined(FMHA_BUILD_ON_GFX950)
  static constexpr bool kTrLoadAvailable = true;
#else
  static constexpr bool kTrLoadAvailable = false;
#endif

#if defined(FMHA_BUILD_ON_GFX950)
  // seq_len runtime threshold for switching fmha_fwd_v3 and qr_async_tr_load
  // pipeline on gfx950.
  // Note: this number need to be tuned if we want to get better performance
  static constexpr int switch_seqlen_threshold = 3000;
  using FmhaV3Shape = typename FmhaFwdSpecificShapeForV3::shape;
  using FmhaQRAsyncTrloadShape =
      typename FmhaFwdSpecificShapeForQRAsyncTrload::shape;
#endif

  template <typename FmhaTraits>
  using AttentionVariant = ck_tile::ComposedAttention<
      FmhaTraits::kHasLogitsSoftCap * ck_tile::LOGITS_SOFT_CAP,
      CK_TILE_FMHA_FWD_FAST_EXP2>;

  template <
      typename FmhaShape,
      typename FmhaTraits,
      typename FmhaMask,
      bool kUseTrLoad>
  using FmhaPipelineProblemTemp = ck_tile::BlockFmhaPipelineProblem<
      typename FmhaFwdTypeConfig<ScalarType>::QDataType,
      typename FmhaFwdTypeConfig<ScalarType>::KDataType,
      typename FmhaFwdTypeConfig<ScalarType>::VDataType,
      typename FmhaFwdTypeConfig<ScalarType>::SaccDataType,
      typename FmhaFwdTypeConfig<ScalarType>::SMPLComputeDataType,
      typename FmhaFwdTypeConfig<ScalarType>::BiasDataType,
      typename FmhaFwdTypeConfig<ScalarType>::RandValOutputDataType,
      typename FmhaFwdTypeConfig<ScalarType>::LSEDataType,
      typename FmhaFwdTypeConfig<ScalarType>::PDataType,
      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
      typename FmhaFwdTypeConfig<ScalarType>::ODataType,
      FmhaShape,
      true, // kIsGroupMode
      AttentionVariant<FmhaTraits>,
      FmhaMask,
      kUseTrLoad,
      FmhaTraits>;

#if defined(FMHA_BUILD_ON_GFX950)
  template <typename FmhaTraits, typename FmhaMask>
  using FmhaPipelineProblemV3Temp = ck_tile::BlockFmhaPipelineProblem<
      typename FmhaFwdTypeConfig<ScalarType>::QDataType,
      typename FmhaFwdTypeConfig<ScalarType>::KDataType,
      typename FmhaFwdTypeConfig<ScalarType>::VDataType,
      typename FmhaFwdTypeConfig<ScalarType>::SaccDataType,
      typename FmhaFwdTypeConfig<ScalarType>::SMPLComputeDataType,
      typename FmhaFwdTypeConfig<ScalarType>::BiasDataType,
      typename FmhaFwdTypeConfig<ScalarType>::RandValOutputDataType,
      typename FmhaFwdTypeConfig<ScalarType>::LSEDataType,
      typename FmhaFwdTypeConfig<ScalarType>::PDataType,
      typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
      typename FmhaFwdTypeConfig<ScalarType>::ODataType,
      FmhaV3Shape,
      true, // kIsGroupMode
      AttentionVariant<FmhaTraits>,
      FmhaMask,
      false,
      FmhaTraits>;

  template <typename FmhaTraits, typename FmhaMask>
  using FmhaPipelineProblemQRAsyncTrloadTemp =
      ck_tile::BlockFmhaPipelineProblem<
          typename FmhaFwdTypeConfig<ScalarType>::QDataType,
          typename FmhaFwdTypeConfig<ScalarType>::KDataType,
          typename FmhaFwdTypeConfig<ScalarType>::VDataType,
          typename FmhaFwdTypeConfig<ScalarType>::SaccDataType,
          typename FmhaFwdTypeConfig<ScalarType>::SMPLComputeDataType,
          typename FmhaFwdTypeConfig<ScalarType>::BiasDataType,
          typename FmhaFwdTypeConfig<ScalarType>::RandValOutputDataType,
          typename FmhaFwdTypeConfig<ScalarType>::LSEDataType,
          typename FmhaFwdTypeConfig<ScalarType>::PDataType,
          typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
          typename FmhaFwdTypeConfig<ScalarType>::ODataType,
          FmhaQRAsyncTrloadShape,
          true, // kIsGroupMode
          AttentionVariant<FmhaTraits>,
          FmhaMask,
          true, // kUseTrLoad
          FmhaTraits>;
#endif

  static void Run(GroupedForwardParams& param, hipStream_t stream) {
    using FmhaMask = ck_tile::SimplifiedGenericAttentionMask<kHasMask>;

    constexpr ck_tile::index_t occupancy = -1;

    constexpr auto kBiasEnum = kHasBias
        ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
        : ck_tile::BlockAttentionBiasEnum::NO_BIAS;

#if defined(FMHA_BUILD_ON_GFX950)
    const bool disable_special_treatment = []() {
      const char* env_p = std::getenv("FMHA_DISABLE_SPECIAL_TREATMENT");
      if (env_p == nullptr)
        return false;
      return static_cast<bool>(atoi(env_p));
    }();

    if (!disable_special_treatment) {
      // only use fmha_fwd_v3 and qr_async_trload pipeline with hdim=128
      if (param.K == 128 && param.Kv == 128) {
        // use fmha_fwd_v3 pipeline if seqlen > switch_seqlen_threshold and mask
        // type is 0(no_mask) or 2(bottom-right casual mask)
        if (param.M > switch_seqlen_threshold &&
            (param.custom_mask_type == 0 || param.custom_mask_type == 2)) {
          if constexpr (MaxK == 128 && !kHasDropout && !kHasBias) {
            using FmhaTraits = ck_tile::TileFmhaTraits<
                false, // kPadSeqLenQ,
                false, // kPadSeqLenK,
                false, // kPadHeadDimQ,
                false, // kPadHeadDimV,
                false, // kHasLogitsSoftCap
                ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                false, // kHasBiasGrad place-holder
                false, // kStoreLSE
                false, // kHasDropout
                ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                1 // Occupancy place-holder(-1 will make the build fail)
                >;
            using FmhaMaskForV3 =
                ck_tile::GenericAttentionMask<kHasMask, false>;
            using FmhaPipelineProblem =
                FmhaPipelineProblemV3Temp<FmhaTraits, FmhaMaskForV3>;
            using FmhaPipeline =
                ck_tile::BlockFmhaFwdV3Pipeline<FmhaPipelineProblem>;
            using FmhaEpilogue =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                    typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                    false,
                    false>>;
            using FmhaKernel =
                ck_tile::FmhaFwdV3Kernel<FmhaPipeline, FmhaEpilogue>;
            RunWithKernelForV3<FmhaKernel>(param, stream);
            // skip the following pipeline if use fmha_fwd_v3 pipeline
            return;
          } else {
            // do nothing, no needs to compile
          }
        } else {
          // use qr_async_trload pipeline if seqlen <= switch_seqlen_threshold
          if constexpr (MaxK == 128) {
            using FmhaTraits = ck_tile::TileFmhaTraits<
                false, // kPadSeqLenQ,
                false, // kPadSeqLenK,
                false, // kPadHeadDimQ,
                false, // kPadHeadDimV,
                false, // kHasLogitsSoftCap
                kBiasEnum,
                false, // kHasBiasGrad place-holder
                false, // kStoreLSE
                kHasDropout,
                ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                1 // Occupancy place-holder(-1 will make the build fail)
                >;
            using FmhaPipelineProblem =
                FmhaPipelineProblemQRAsyncTrloadTemp<FmhaTraits, FmhaMask>;
            using FmhaPipeline = ck_tile::BlockFmhaPipelineQRKSVSAsyncTrload<
                FmhaPipelineProblem>;
            using FmhaEpilogue =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                    typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                    false,
                    false>>;
            using FmhaKernel =
                ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;
            RunWithKernel<FmhaKernel>(param, stream);
            return;
          } else {
            // do nothing, no needs to compile
          }
        }
      } else {
        // do nothing, go to the following pipeline selection
      }
    };
#endif

    // no need to check seqlen_q since it is not used as fastest dim,
    // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
    constexpr bool kPadSeqLenQ = false;
    constexpr bool kPadSeqLenK = true;

    const bool enable_async_pipeline = []() {
      const char* env_p = std::getenv("FMHA_ENABLE_ASYNC_PIPELINE");
      if (env_p == nullptr)
        return false;
      return static_cast<bool>(atoi(env_p));
    }();

    const bool use_async_pipeline =
        (!kHasBias && (param.K % 8 == 0) && (param.Kv % 8 == 0) &&
         (MaxK <= 128 && MTile <= 128));

    if (!(use_async_pipeline && enable_async_pipeline)) {
      using FmhaShape = typename std::conditional_t<
          kUseWholeKPrefetchPipeline,
          FmhaFwdWholeKPrefetchShape<MaxK, MTile>,
          FmhaFwdCommonShape<MaxK, MTile>>::Type;

      const bool pad_headdim_v = !(param.Kv % FmhaShape::kN1 == 0);

      const bool pad_headdim_q = [&]() {
        // qr_ks_vs_whole_k_prefetch pipeline naively support hdim96/hdim160
        if constexpr (kUseWholeKPrefetchPipeline)
          return !(param.K % FmhaShape::kQKHeaddim == 0);
        else
          return !(param.K % FmhaShape::kSubQKHeaddim == 0);
      }();

      BOOL_SWITCH_2(
          pad_headdim_q, kPadHeadDimQ, pad_headdim_v, kPadHeadDimV, [&] {
            using FmhaTraits = ck_tile::TileFmhaTraits<
                kPadSeqLenQ,
                kPadSeqLenK,
                kPadHeadDimQ,
                kPadHeadDimV,
                false, // kHasLogitsSoftCap
                kBiasEnum,
                false, // kHasBiasGrad place-holder
                false, // kStoreLSE
                kHasDropout,
                ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                occupancy>;

            using FmhaEpilogue =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                    typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                    kPadSeqLenQ,
                    kPadHeadDimV>>;

            if constexpr (kUseWholeKPrefetchPipeline) {
              using FmhaPipelineProblem = std::conditional_t<
                  kTrLoadAvailable,
                  FmhaPipelineProblemTemp<
                      FmhaShape,
                      FmhaTraits,
                      FmhaMask,
                      true>,
                  FmhaPipelineProblemTemp<
                      FmhaShape,
                      FmhaTraits,
                      FmhaMask,
                      false>>;
              using FmhaPipeline = std::conditional_t<
                  kTrLoadAvailable,
                  ck_tile::BlockFmhaPipelineQRKSVSWholeKPrefetchTrLoad<
                      FmhaPipelineProblem>,
                  ck_tile::BlockFmhaPipelineQRKSVSWholeKPrefetch<
                      FmhaPipelineProblem>>;
              using FmhaKernel =
                  ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

              RunWithKernel<FmhaKernel>(param, stream);
            } else if constexpr (MaxK <= 256) {
              using FmhaPipelineProblem = FmhaPipelineProblemTemp<
                  FmhaShape,
                  FmhaTraits,
                  FmhaMask,
                  false>;
              using FmhaPipeline =
                  ck_tile::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;
              using FmhaKernel =
                  ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

              RunWithKernel<FmhaKernel>(param, stream);
            } else {
              using FmhaPipelineProblem = FmhaPipelineProblemTemp<
                  FmhaShape,
                  FmhaTraits,
                  FmhaMask,
                  false>;
              using FmhaPipeline =
                  ck_tile::BlockFmhaPipelineQSKSVS<FmhaPipelineProblem>;
              using FmhaKernel =
                  ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

              RunWithKernel<FmhaKernel>(param, stream);
            }
          });
    } else {
      if constexpr (MaxK <= 128 && MTile <= 128) {
        using FmhaShape = typename FmhaFwdCommonShape<MaxK, MTile>::Type;

        using FmhaTraits = ck_tile::TileFmhaTraits<
            true, // kPadSeqLenQ,
            kPadSeqLenK,
            true, // kPadHeadDimQ,
            true, // kPadHeadDimV,
            false, // kHasLogitsSoftCap
            kBiasEnum,
            false, // kHasBiasGrad place-holder
            false, // kStoreLSE
            kHasDropout,
            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
            occupancy>;

        using FmhaPipelineProblem =
            FmhaPipelineProblemTemp<FmhaShape, FmhaTraits, FmhaMask, false>;

        using FmhaPipeline =
            ck_tile::BlockFmhaPipelineQRKSVSAsync<FmhaPipelineProblem>;

        using FmhaEpilogue =
            ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                typename FmhaFwdTypeConfig<ScalarType>::OaccDataType,
                typename FmhaFwdTypeConfig<ScalarType>::ODataType,
                true,
                true>>;

        using FmhaKernel = ck_tile::FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

        RunWithKernel<FmhaKernel>(param, stream);
      } else {
        /* runtime will never get here, so no codes to compile */
      };
    };
  };

  template <typename FmhaKernel>
  static void RunWithKernel(GroupedForwardParams& param, hipStream_t stream) {
    const auto kargs = FmhaKernel::MakeKargs(
        param.q_ptr,
        param.k_ptr,
        param.v_ptr,
        param.attn_bias_ptr,
        nullptr, // q_descale_ptr
        nullptr, // k_descale_ptr
        nullptr, // v_descale_ptr
        nullptr, // rand_val_ptr
        nullptr, // lse_ptr
        param.out_ptr,
        param.seqstart_q_dev_ptr,
        param.seqstart_k_dev_ptr,
        nullptr, // seqlen_q_ptr, most recently added kernel argument
        param.seqlen_k_dev_ptr,
        nullptr, // block_scale_seqstart_q_ptr
        nullptr, // block_scale_seqstart_k_ptr
        nullptr, // seqstart_v_scale_ptr
        param.K, // hdim_q
        param.Kv, // hdim_v
        param.Hq, // nhead_q
        param.Hq / param.Hkv, // nhead_ratio_qk
        param.scale,
        0.f, // logits_soft_cap
        param.q_strides[0], // q, k, v, bias, randval, out tensor seq-dim
                            // stride
        param.k_strides[0],
        param.v_strides[0],
        param.attn_bias_strides[2],
        0, // stride_randval
        param.out_strides[0],
        0, // stride_q_descale
        0, // stride_k_descale
        0, // stride_v_descale
        param.q_strides[1], // q, k, v, bias, randval, lse, out tensor
                            // head-dim stride
        param.k_strides[1],
        param.v_strides[1],
        param.attn_bias_strides[1],
        0, // nhead_stride_randval
        0, // nhead_stride_lse
        param.out_strides[1],
        0, // nhead_stride_q_descale
        0, // nhead_stride_k_descale
        0, // nhead_stride_v_descale
        (param.window_size > 0) ? param.window_size - 1
                                : -1, // window_left_size
        (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
        0, // sink size
        param.custom_mask_type,
        0, // min_seqlen_q, most recently added kernel argument
        param.dropout_prob,
        false, // is_store_randval
        std::make_pair(param.philox_seed, param.philox_offset),
        0, // block_scale_size_q
        0); // block_scale_size_kv

    dim3 kGridSize = FmhaKernel::GridSize(
        param.num_batches,
        param.Hq,
        param.max_seqlen_q,
        param.Kv,
        param.seqlen_k_dev_ptr != nullptr);
    dim3 kBlockSize = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            FmhaKernel{}, kGridSize, kBlockSize, 0, kargs));
  };

#if defined(FMHA_BUILD_ON_GFX950)
  template <typename FmhaKernel>
  static void RunWithKernelForV3(
      GroupedForwardParams& param,
      hipStream_t stream) {
    /// NOTICE: This was borrowed from Aiter. Make sure the selected remap_opt
    /// setting truly maximizes the kernel's performance.
    int remap_opt = 2;
    if (!param.custom_mask_type && ((param.Hq % 8 != 0) || (param.M > 16384))) {
      if (param.M >= 65536) {
        remap_opt = 0;
      } else {
        remap_opt = 1;
      }
    }
    const auto kargs = FmhaKernel::MakeKargs(
        param.q_ptr,
        param.k_ptr,
        param.v_ptr,
        nullptr, // lse_ptr
        param.out_ptr,
        param.seqstart_q_dev_ptr,
        param.seqstart_k_dev_ptr,
        nullptr, // seqlen_q_ptr
        param.seqlen_k_dev_ptr,
        param.K, // hdim_q
        param.Kv, // hdim_v
        param.Hq, // nhead_q
        param.Hq / param.Hkv, // nhead_ratio_qk
        param.scale,
        0.0f, // logits_soft_cap
        param.q_strides[0], // q, k, v, out tensor seq-dim
                            // stride
        param.k_strides[0],
        param.v_strides[0],
        param.out_strides[0],
        param.q_strides[1], // q, k, v, lse, out tensor
                            // head-dim stride
        param.k_strides[1],
        param.v_strides[1],
        0, // nhead_stride_lse
        param.out_strides[1],
        (param.window_size > 0) ? param.window_size - 1
                                : -1, // window_left_size
        (param.custom_mask_type == 0) ? -1 : 0, // window_right_size
        param.custom_mask_type,
        remap_opt);

    dim3 kGridSize = FmhaKernel::GridSize(
        param.num_batches, param.Hq, param.max_seqlen_q, param.Kv);
    dim3 kBlockSize = FmhaKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            FmhaKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
#endif
};
