# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, List, Optional, Tuple

import torch

from xformers.ops.common import register_operator
from xformers.ops.fmha import ck as ck_fmha
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from xformers.ops.fmha.common import (
    check_lastdim_alignment_stride1,
    Context,
    Inputs,
)


@register_operator
class FwOp(ck_fmha.FwOp):

    OPERATOR = ck_fmha.FwOp.OPERATOR
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }
    SUPPORTED_MAX_K = 256
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "ck_splitKF"

    SPLIT_K: Optional[int] = None
    BLOCK_M = 16
    BLOCK_N = 64

    NUM_GROUPS = 1  # Default quantization is row-wise

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        # if K not in {16, 32, 64, 128}:
        #     reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super().not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
        check_lastdim_alignment_stride1(reasons, "value", d.value, 8)

        q_len = d.query.shape[1]
        if isinstance(d.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            seqinfo = d.attn_bias.q_seqinfo
            if q_len != seqinfo.seqstart_py[-1]:
                reasons.append(
                    f"Expected total {seqinfo.seqstart_py[-1]} queries not {q_len}"
                )
            q_len = seqinfo.min_seqlen
            if q_len != seqinfo.max_seqlen:
                reasons.append(
                    "Variable query len is not supported in the presence of causal mask."
                )

        if d.key.ndim in [4, 5] and d.key.shape[-2] != 1:
            if d.key.stride(-2) == 0 and d.value.stride(-2) == 0 and q_len > 1:
                reasons.append("multiquery is only supported with query seqlen=1")

        return reasons

    @classmethod
    def get_split_k(cls, B: int, H: int, Mk: int) -> int:
        """Heuristic for the number of splits"""
        bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
        split_k = max(Mk, 1024) // bh
        max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
        while split_k > 0 and Mk / split_k < max_chunk_size:
            split_k = split_k // 2
        split_k = min(split_k, 64)
        split_k = max(split_k, 1)
        return split_k

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        return ck_fmha.FwOp.apply_with_num_kv_splits(
            inp, needs_gradient=needs_gradient, num_kv_splits=cls.SPLIT_K
        )


class FwOp_S1(FwOp):
    SPLIT_K = 1
    NAME = "ck_splitK1"


class FwOp_S2(FwOp):
    SPLIT_K = 2
    NAME = "ck_splitK2"


class FwOp_S4(FwOp):
    SPLIT_K = 4
    NAME = "ck_splitK4"


class FwOp_S8(FwOp):
    SPLIT_K = 8
    NAME = "ck_splitK8"


class FwOp_S16(FwOp):
    SPLIT_K = 16
    NAME = "ck_splitK16"


class FwOp_S32(FwOp):
    SPLIT_K = 32
    NAME = "ck_splitK32"


class FwOp_S64(FwOp):
    SPLIT_K = 64
    NAME = "ck_splitK64"


class FwOp_S128(FwOp):
    SPLIT_K = 128
    NAME = "ck_splitK128"
