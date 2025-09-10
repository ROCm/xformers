/*
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

struct FmhaFwdSpecificConfig {
  int B; // batch size
  int M; // seq_len for Query
  int N; // seq_len for Key and Value
  int Hq; // number of heads for Query
  int Hkv; // number of heads for Key and Value
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  std::optional<bool> kHasMask; // whether the attention mask is used
};

// bf16 specific configs for using fmha_fwd_v3_pipeline on gfx950
static const std::vector<FmhaFwdSpecificConfig>
    g_qr_async_tr_load_pipeline_bf16_configs = {
        // clang format off
        //  |    B    |    M    |    N    |    Hq   |   Hkv  |    K   |   Kv  |
           {     1,      2048,     2048,       8,       8,       128,    128   },
           {     1,      2048,     2048,      15,       1,       128,    128   }
        // clang format on
};

// bf16 specific configs for using fmha_fwd_v3_pipeline on gfx950
// Note: fmha_fwd_v3_pipeline has somthing wrong with kHasMask=true case, so we
// only support kHasMask=false case now
static const std::vector<FmhaFwdSpecificConfig>
    g_fmha_fwd_v3_pipeline_bf16_specific_configs = {
        // clang format off
        //  |    B    |    M    |    N    |    Hq   |   Hkv  |    K   |   Kv  |   kHasMask    |
           {     1,      16384,    16384,      8,        8,      128,    128,     false        },
           {     1,      16384,    16384,     15,        1,      128,    128,     false        }
        // clang format on
};
