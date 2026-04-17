/*
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>

#if defined(FMHA_BUILD_ON_GFX11) || defined(FMHA_BUILD_ON_GFX12)
// RDNA WMMA only registers 16x16x16 in CK tile's warp_gemm dispatcher for
// fp16 / bf16 (and fp8 / int8). The xformers wrapper layer was authored for
// CDNA MFMA shapes (32x32x16 and 16x16x32). To keep the per-arch *_setting.h
// headers small, we alias all three names down to the WMMA-supported shape.
// Block tile geometry is unchanged — each warp simply issues more WMMA
// instructions to cover the same output tile.
using WarpTile_32x32x16 = ck_tile::sequence<16, 16, 16>;
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 16>;
using WarpTile_16x16x16 = ck_tile::sequence<16, 16, 16>;
#else
using WarpTile_32x32x16 = ck_tile::sequence<32, 32, 16>;
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 32>;
using WarpTile_16x16x16 = ck_tile::sequence<16, 16, 16>;
#endif
