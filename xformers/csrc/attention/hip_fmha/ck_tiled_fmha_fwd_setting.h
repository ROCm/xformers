/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha.hpp>
#include "ck_fmha_util.h"
#include "ck_tiled_fmha_fwd_type_config.h"
#include "ck_tiled_fmha_warp_tile_define.h"

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct FmhaFwdBlockTile;

// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
//
template <ck_tile::index_t MTile>
struct FmhaFwdBlockTile<32, MTile> {
  using type = ck_tile::sequence<64, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdBlockTile<64, MTile> {
  using type = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdBlockTile<96, MTile> {
  using type = ck_tile::sequence<128, 128, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdBlockTile<128, 64> {
  using type = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdBlockTile<128, 128> {
  using type = ck_tile::sequence<128, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdBlockTile<256, MTile> {
  using type = ck_tile::sequence<128, 128, 32, 256, 32, 256>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdBlockTile<512, MTile> {
  using type = ck_tile::sequence<64, 128, 32, 512, 32, 512>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MaxK, ck_tile::index_t MTile>
struct FmhaFwdShape;

template <ck_tile::index_t MTile>
struct FmhaFwdShape<32, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdBlockTile<32>::type,
      typename FmhaFwdBlockTile<32>::gemm0_warps,
      WarpTile_32x32x16,
      typename FmhaFwdBlockTile<32>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdShape<32, 64>;
template struct FmhaFwdShape<32, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdShape<64, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdBlockTile<64>::type,
      typename FmhaFwdBlockTile<64>::gemm0_warps,
      WarpTile_32x32x16,
      typename FmhaFwdBlockTile<64>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdShape<64, 64>;
template struct FmhaFwdShape<64, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdShape<96, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdBlockTile<96>::type,
      typename FmhaFwdBlockTile<96>::gemm0_warps,
      WarpTile_32x32x16,
      typename FmhaFwdBlockTile<96>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdShape<96, 64>;
template struct FmhaFwdShape<96, 128>;

template <>
struct FmhaFwdShape<128, 64> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdBlockTile<128, 64>::type,
      typename FmhaFwdBlockTile<128, 64>::gemm0_warps,
      WarpTile_16x16x32,
      typename FmhaFwdBlockTile<128, 64>::gemm1_warps,
      WarpTile_16x16x16,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdShape<128, 128> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdBlockTile<128, 128>::type,
      typename FmhaFwdBlockTile<128, 128>::gemm0_warps,
      WarpTile_32x32x16,
      typename FmhaFwdBlockTile<128, 128>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdShape<256, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdBlockTile<256>::type,
      typename FmhaFwdBlockTile<256>::gemm0_warps,
      WarpTile_32x32x16,
      typename FmhaFwdBlockTile<256>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdShape<256, 64>;
template struct FmhaFwdShape<256, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdShape<512, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename FmhaFwdBlockTile<512>::type,
      typename FmhaFwdBlockTile<512>::gemm0_warps,
      WarpTile_16x16x16,
      typename FmhaFwdBlockTile<512>::gemm1_warps,
      WarpTile_16x16x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdShape<512, 64>;
template struct FmhaFwdShape<512, 128>;

// Specific shape for fmha_fwd_v3_pipeline and qr_async_tr_load_pipeline on
// gfx950
#if defined(FMHA_BUILD_ON_GFX950)
struct FmhaFwdSpecificShapeForV3 {
  using block_tile = ck_tile::sequence<256, 32, 128, 128, 32, 128>;
  using gemm_warps = ck_tile::sequence<8, 1, 1>;

  using shape = ck_tile::TileFmhaShape<
      block_tile,
      gemm_warps,
      WarpTile_32x32x16,
      gemm_warps,
      WarpTile_32x32x16,
      true // IsVLayoutRowMajor
      >;
};

struct FmhaFwdSpecificShapeForQRAsyncTrload {
  using block_tile = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm_warps = ck_tile::sequence<4, 1, 1>;

  using shape = ck_tile::TileFmhaShape<
      block_tile,
      gemm_warps,
      WarpTile_16x16x32,
      gemm_warps,
      WarpTile_16x16x32,
      true // IsVLayoutRowMajor
      >;
};
#endif

static int get_fmha_fwd_mtile(
    int num_batches,
    int num_heads,
    int max_seqlen_q) {
  int num_SMs = get_number_of_cu();
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  int batch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, 128);

  if (batch_nhead_mblocks >= 0.8 * num_SMs)
    return 128;

  // currently, only hdim-128 can use mtile-64, for other hdim, the settings for
  // mtile-64 can be added through tuning/verification
  return 64;
};

static int get_fmha_fwd_least_mtile() {
  return 64;
};
