/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <optional>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <ck/utility/common_header.hpp>
#include <ck/host_utility/device_prop.hpp>
#include <ck/host_utility/kernel_launch.hpp>
#include <ck/tensor_description/cluster_descriptor.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <ck/tensor/tensor_view.hpp>

#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp>
#include <ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qs_ks_vs.hpp>
#include <ck/tile_program/tile/tile_fmha_shape.hpp>
#include <ck/tile_program/tile/tile_fmha_traits.hpp>
#include <ck/tile_program/block_tile/block_masking.hpp>

#include "ck_tiled_fmha_forward_kernel.h"
#include "ck_tiled_fmha_fwd_epilogue.h"
#include "ck_tiled_fmha_fwd_tile_partitioner.h"
#include "ck_tiled_fmha_params.h"
#include "ck_tiled_fmha_definitions.h"

#include "ck_tiled_bool_switch.h"

template <typename scalar_t, bool has_causal_mask, bool has_attn_bias>
struct grouped_infer_causalmask_attnbias_dispatched
{
    using FmhaEpilogue =
        FmhaFwdEpilogue<FmhaFwdEpilogueProblem<typename FmhaFwdTypeConfig<scalar_t>::OaccDataType,
                                               typename FmhaFwdTypeConfig<scalar_t>::ODataType>>;

#ifndef GROUPED_INFER_HEADDIM_SWITCH
#define GROUPED_INFER_HEADDIM_SWITCH(HEAD_DIM1, HEAD_DIM2, CONST_NAME, ...) \
    [&] {                                                                   \
        if(HEAD_DIM1 <= 32 && HEAD_DIM2 <= 32)                              \
        {                                                                   \
            constexpr ck::index_t CONST_NAME = 32;                          \
            __VA_ARGS__();                                                  \
        }                                                                   \
        else if(HEAD_DIM1 <= 64 && HEAD_DIM2 <= 64)                         \
        {                                                                   \
            constexpr ck::index_t CONST_NAME = 64;                          \
            __VA_ARGS__();                                                  \
        }                                                                   \
        else if(HEAD_DIM1 <= 128 && HEAD_DIM2 <= 128)                       \
        {                                                                   \
            constexpr ck::index_t CONST_NAME = 128;                         \
            __VA_ARGS__();                                                  \
        }                                                                   \
        else if(HEAD_DIM1 <= 256 && HEAD_DIM2 <= 256)                       \
        {                                                                   \
            constexpr ck::index_t CONST_NAME = 256;                         \
            __VA_ARGS__();                                                  \
        }                                                                   \
        else                                                                \
        {                                                                   \
            throw std::runtime_error("Head-dim sizes not supported!");      \
        }                                                                   \
    }()
#endif

    template <typename FmhaTraits, ck::index_t HDim, typename FmhaMask>
    using FmhaPipelineProblemTemp = ck::tile_program::block::BlockFmhaPipelineProblem<
        typename FmhaFwdTypeConfig<scalar_t>::QDataType,
        typename FmhaFwdTypeConfig<scalar_t>::KDataType,
        typename FmhaFwdTypeConfig<scalar_t>::VDataType,
        typename FmhaFwdTypeConfig<scalar_t>::SaccDataType,
        typename FmhaFwdTypeConfig<scalar_t>::SMPLComputeDataType,
        typename FmhaFwdTypeConfig<scalar_t>::BiasDataType,
        typename FmhaFwdTypeConfig<scalar_t>::LSEDataType,
        typename FmhaFwdTypeConfig<scalar_t>::PDataType,
        typename FmhaFwdTypeConfig<scalar_t>::OaccDataType,
        typename FmhaFwdTypeConfig<scalar_t>::ODataType,
        HDim == 32 ? 128 : 256, // BlockSize
        FmhaFwdShape<HDim>,
        true, // kIsGroupMode
        FmhaMask,
        FmhaTraits>;

    static void Run(GroupedForwardParams& param, hipStream_t stream)
    {
        const bool has_local_attention = (param.window_size > 0) ? true : false;

        BOOL_SWITCH(has_local_attention, USE_LOCAL_ATTENTION, [&] {
            constexpr bool has_masking = has_causal_mask || USE_LOCAL_ATTENTION;

            using FmhaMask =
                ck::tile_program::block::GenericAttentionMask<has_masking, USE_LOCAL_ATTENTION>;

            GROUPED_INFER_HEADDIM_SWITCH(param.K, param.Kv, HDim, [&] {
                using FmhaShape                 = FmhaFwdShape<HDim>;
                using FmhaTilePartitioner       = FmhaFwdTilePartitioner<FmhaShape>;
                constexpr ck::index_t occupancy = (HDim == 64) ? 3 : 2;

                bool k0n1_need_padding =
                    !(param.K % FmhaShape::kK0BlockLength == 0 && param.Kv % FmhaShape::kN1 == 0);

                constexpr bool kM0NeedPadding   = true;
                constexpr bool kN0K1NeedPadding = true;

                BOOL_SWITCH(k0n1_need_padding, kK0N1NeedPadding, [&] {
                    using FmhaTraits = ck::tile_program::TileFmhaTraits<kM0NeedPadding,
                                                                        kN0K1NeedPadding,
                                                                        kK0N1NeedPadding,
                                                                        has_attn_bias,
                                                                        false, // kStoreLSE
                                                                        occupancy>;

                    using FmhaPipelineProblem = FmhaPipelineProblemTemp<FmhaTraits, HDim, FmhaMask>;

                    if constexpr(HDim == 256)
                    {
                        using FmhaPipeline =
                            ck::tile_program::block::BlockFmhaPipelineQSKSVS<FmhaPipelineProblem>;
                        using FmhaKernel =
                            FmhaFwdKernel<FmhaTilePartitioner, FmhaPipeline, FmhaEpilogue>;

                        RunWithKernel<FmhaKernel>(param, stream);
                    }
                    else
                    {
                        using FmhaPipeline =
                            ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;
                        using FmhaKernel =
                            FmhaFwdKernel<FmhaTilePartitioner, FmhaPipeline, FmhaEpilogue>;

                        RunWithKernel<FmhaKernel>(param, stream);
                    }
                });
            });
        });
    };

    template <typename FmhaKernel>
    static void RunWithKernel(GroupedForwardParams& param, hipStream_t stream)
    {
        const auto kargs = [&] {
            return FmhaKernel::MakeKargs(
                param.q_ptr,
                param.k_ptr,
                param.v_ptr,
                param.attn_bias_ptr,
                nullptr, // lse_ptr
                param.out_ptr,
                param.seqstart_q_dev_ptr,
                param.seqstart_k_dev_ptr,
                param.seqlen_k_dev_ptr,
                param.K,              // hdim_q
                param.Kv,             // hdim_v
                param.Hq / param.Hkv, // nhead_ratio_qk
                param.scale,
                param.q_strides[0], // q, k, v, bias, out tensor seq-dim stride
                param.k_strides[0],
                param.v_strides[0],
                param.attn_bias_strides[2],
                param.out_strides[0],
                param.q_strides[1], // q, k, v, bias, lse, out tensor head-dim stride
                param.k_strides[1],
                param.v_strides[1],
                param.attn_bias_strides[1],
                0, // nhead_stride_lse
                param.out_strides[1],
                static_cast<CausalMaskType>(param.custom_mask_type),
                param.window_size);
        }();

        dim3 kGridSize =
            FmhaKernel::GridSize(param.num_batches, param.Hq, param.max_seqlen_q, param.Kv);
        constexpr dim3 kBlockSize         = FmhaKernel::BlockSize();
        constexpr ck::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

        (void)launch_kernel<kBlockSize.x, kBlockPerCu>(
            StreamConfig{stream, false}, FmhaKernel{}, kGridSize, kBlockSize, 0, kargs);
    };
};

template <typename scalar_t, bool has_causal_mask, bool has_attn_bias>
void run_grouped_infer_causalmask_attnbias_dispatched(GroupedForwardParams& param,
                                                      hipStream_t stream)
{
    grouped_infer_causalmask_attnbias_dispatched<scalar_t, has_causal_mask, has_attn_bias>::Run(
        param, stream);
};
