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

namespace detail {

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct FmhaFwdCommonBlockTile;

template <ck_tile::index_t MaxK, ck_tile::index_t MTile = 0>
struct FmhaFwdWholeKPrefetchBlockTile;

// tile settings used for gfx908/gfx90a/gfx942
#if !defined(FMHA_BUILD_ON_GFX950)
// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<32, MTile> {
  using tile_lengths = ck_tile::sequence<64, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<64, MTile> {
  using tile_lengths = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<96, MTile> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdCommonBlockTile<128, 64> {
  using tile_lengths = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdCommonBlockTile<128, 128> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<256, MTile> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 256, 32, 256>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<512, MTile> {
  using tile_lengths = ck_tile::sequence<64, 128, 32, 512, 32, 512>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

// Tile-sizes: M N0 K0/N0Sub N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchBlockTile<32, MTile> {
  using tile_lengths = ck_tile::sequence<64, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchBlockTile<64, MTile> {
  using tile_lengths = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchBlockTile<96, MTile> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdWholeKPrefetchBlockTile<128, 64> {
  using tile_lengths = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdWholeKPrefetchBlockTile<128, 128> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};
#endif

// tile settings used for gfx950
#if defined(FMHA_BUILD_ON_GFX950)
// Tile-sizes: M N0 K0 N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<32, MTile> {
  using tile_lengths = ck_tile::sequence<64, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<64, MTile> {
  using tile_lengths = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<96, MTile> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdCommonBlockTile<128, 64> {
  using tile_lengths = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdCommonBlockTile<128, 128> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<256, MTile> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 256, 32, 256>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonBlockTile<512, MTile> {
  using tile_lengths = ck_tile::sequence<64, 128, 32, 512, 32, 512>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

// Tile-sizes: M N0 K0/N0Sub N1 K1 MaxK (MaxK % K0 == 0, MaxK % N1 == 0, N0 % K1 == 0)
template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchBlockTile<32, MTile> {
  using tile_lengths = ck_tile::sequence<64, 64, 16, 32, 32, 32>;
  using gemm0_warps = ck_tile::sequence<2, 1, 1>;
  using gemm1_warps = ck_tile::sequence<2, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchBlockTile<64, MTile> {
  using tile_lengths = ck_tile::sequence<128, 64, 32, 64, 32, 64>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchBlockTile<96, MTile> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 96>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdWholeKPrefetchBlockTile<128, 64> {
  using tile_lengths = ck_tile::sequence<64, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};

template <>
struct FmhaFwdWholeKPrefetchBlockTile<128, 128> {
  using tile_lengths = ck_tile::sequence<128, 128, 32, 128, 32, 128>;
  using gemm0_warps = ck_tile::sequence<4, 1, 1>;
  using gemm1_warps = ck_tile::sequence<4, 1, 1>;
};
#endif

}; // namespace detail

template <ck_tile::index_t MaxK, ck_tile::index_t MTile>
struct FmhaFwdCommonShape;

template <ck_tile::index_t MaxK, ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchShape;

#if !defined(FMHA_BUILD_ON_GFX950)
template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<32, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<32>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<32>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<32>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<32, 64>;
template struct FmhaFwdCommonShape<32, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<64, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<64>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<64>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<64>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<64, 64>;
template struct FmhaFwdCommonShape<64, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<96, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<96>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<96>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<96>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<96, 64>;
template struct FmhaFwdCommonShape<96, 128>;

template <>
struct FmhaFwdCommonShape<128, 64> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<128, 64>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<128, 64>::gemm0_warps,
      WarpTile_16x16x32,
      typename detail::FmhaFwdCommonBlockTile<128, 64>::gemm1_warps,
      WarpTile_16x16x16,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdCommonShape<128, 128> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<128, 128>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<128, 128>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<128, 128>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<256, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<256>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<256>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<256>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<256, 64>;
template struct FmhaFwdCommonShape<256, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<512, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<512>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<512>::gemm0_warps,
      WarpTile_16x16x16,
      typename detail::FmhaFwdCommonBlockTile<512>::gemm1_warps,
      WarpTile_16x16x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<512, 64>;
template struct FmhaFwdCommonShape<512, 128>;

// need special consideration when using qr_ks_vs_whole_k_prefetch pipeline
template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchShape<32, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<32>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<32>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<32>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdWholeKPrefetchShape<32, 64>;
template struct FmhaFwdWholeKPrefetchShape<32, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchShape<64, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<64>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<64>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<64>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdWholeKPrefetchShape<64, 64>;
template struct FmhaFwdWholeKPrefetchShape<64, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchShape<96, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<96>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<96>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<96>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdWholeKPrefetchShape<96, 64>;
template struct FmhaFwdWholeKPrefetchShape<96, 128>;

template <>
struct FmhaFwdWholeKPrefetchShape<128, 64> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 64>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 64>::gemm0_warps,
      WarpTile_16x16x32,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 64>::gemm1_warps,
      WarpTile_16x16x16,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdWholeKPrefetchShape<128, 128> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 128>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 128>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 128>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};
#endif

#if defined(FMHA_BUILD_ON_GFX950)
template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<32, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<32>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<32>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<32>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<32, 64>;
template struct FmhaFwdCommonShape<32, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<64, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<64>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<64>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<64>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<64, 64>;
template struct FmhaFwdCommonShape<64, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<96, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<96>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<96>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<96>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<96, 64>;
template struct FmhaFwdCommonShape<96, 128>;

template <>
struct FmhaFwdCommonShape<128, 64> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<128, 64>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<128, 64>::gemm0_warps,
      WarpTile_16x16x32,
      typename detail::FmhaFwdCommonBlockTile<128, 64>::gemm1_warps,
      WarpTile_16x16x32,
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdCommonShape<128, 128> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<128, 128>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<128, 128>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<128, 128>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<256, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<256>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<256>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdCommonBlockTile<256>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<256, 64>;
template struct FmhaFwdCommonShape<256, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdCommonShape<512, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdCommonBlockTile<512>::tile_lengths,
      typename detail::FmhaFwdCommonBlockTile<512>::gemm0_warps,
      WarpTile_16x16x16,
      typename detail::FmhaFwdCommonBlockTile<512>::gemm1_warps,
      WarpTile_16x16x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdCommonShape<512, 64>;
template struct FmhaFwdCommonShape<512, 128>;

// need special consideration when using qr_ks_vs_whole_k_prefetch pipeline
template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchShape<32, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<32>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<32>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<32>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdWholeKPrefetchShape<32, 64>;
template struct FmhaFwdWholeKPrefetchShape<32, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchShape<64, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<64>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<64>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<64>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdWholeKPrefetchShape<64, 64>;
template struct FmhaFwdWholeKPrefetchShape<64, 128>;

template <ck_tile::index_t MTile>
struct FmhaFwdWholeKPrefetchShape<96, MTile> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<96>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<96>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<96>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};

template struct FmhaFwdWholeKPrefetchShape<96, 64>;
template struct FmhaFwdWholeKPrefetchShape<96, 128>;

template <>
struct FmhaFwdWholeKPrefetchShape<128, 64> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 64>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 64>::gemm0_warps,
      WarpTile_16x16x32,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 64>::gemm1_warps,
      WarpTile_16x16x16, // ToDo: 16x16x32 not good
      IsVLayoutRowMajor>;
};

template <>
struct FmhaFwdWholeKPrefetchShape<128, 128> {
  using Type = ck_tile::TileFmhaShape<
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 128>::tile_lengths,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 128>::gemm0_warps,
      WarpTile_32x32x16,
      typename detail::FmhaFwdWholeKPrefetchBlockTile<128, 128>::gemm1_warps,
      WarpTile_32x32x16,
      IsVLayoutRowMajor>;
};
#endif

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
