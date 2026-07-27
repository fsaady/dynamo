# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.unified,
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None,
        reason="tensorrt_llm not installed in this container",
    ),
]


def test_get_disagg_node_id_uses_legacy_modulus():
    from dynamo.trtllm.backend_args import DEFAULT_DISAGG_NODE_ID_MODULUS
    from dynamo.trtllm.utils.disagg_utils import get_disagg_node_id

    assert get_disagg_node_id(1020, DEFAULT_DISAGG_NODE_ID_MODULUS) == 1020
    assert get_disagg_node_id(1021, DEFAULT_DISAGG_NODE_ID_MODULUS) == 0


def test_parse_args_rejects_invalid_disagg_node_id_modulus():
    from dynamo.trtllm.args import parse_args

    with pytest.raises(ValueError, match="--disagg-node-id-modulus"):
        parse_args(["--disagg-node-id-modulus", "0"])
