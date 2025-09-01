/*
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

struct FmhaFwdV3PipelineSpecificConfig {
    std::string device_name;

    int B;      // batch size
    int M;      // seq_len for Query
    int N;      // seq_len for Key and Value
    int Hq;     // number of heads for Query
    int Hkv;    // number of heads for Key and Value
    int K;      // embed_dim for Query and Key
    int Kv;     // embed_dim for Value

    bool kHasMask; // whether the attention mask is used

    FmhaFwdV3PipelineSpecificConfig(
        const std::string& dev,
        int b, int m, int n, int hq, int hkv, int k, int kv,
        bool has_mask)
        : device_name(dev), B(b), M(m), N(n), Hq(hq), Hkv(hkv),
          K(k), Kv(kv), kHasMask(has_mask) {}
};

// bf16 specific configs for using fmha_fwd_v3_pipeline
static std::vector<FmhaFwdV3PipelineSpecificConfig> g_fmha_fwd_v3_pipeline_bf16_specific_configs = {
    // clang format off
    // | device_name |  B  |  M  |  N  | Hq | Hkv |  K  | Kv |    kHasMask    |
    {     "gfx950",     1,   2048,   2048,  8,    8,   128, 128,       false   },
    {     "gfx950",     1,   16384, 16384,  8,    8,   128, 128,       false   },
    {     "gfx950",     1,   2048,   2048, 15,    1,   128, 128,       false   },
    {     "gfx950",     1,   16384, 16384, 15,    1,   128, 128,       false   },
    // clang format on
};

// fp16 specific configs for using fmha_fwd_v3_pipeline
static std::vector<FmhaFwdV3PipelineSpecificConfig> g_fmha_fwd_v3_pipeline_fp16_specific_configs = {
    // clang format off
    // | device_name |  B  |  M  |  N  | Hq | Hkv |  K  | Kv |    kHasMask    |
    {     "gfx950",     1,   2048,   2048,  8,    8,   128, 128,       false   },
    {     "gfx950",     1,   16384, 16384,  8,    8,   128, 128,       false   },
    {     "gfx950",     1,   2048,   2048, 15,    1,   128, 128,       false   },
    {     "gfx950",     1,   16384, 16384, 15,    1,   128, 128,       false   },
    // clang format on
};
