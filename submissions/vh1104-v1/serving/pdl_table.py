TARGETS = {
    'vllm/models/qwen4_exp/nvidia/ops/qsa.py': (
        '8327f0899e47b2c65d95b3f46bdb2db8fb09110f04c5c178850d0c256e22bc04',
        (
            (
                'from vllm.triton_utils import HAS_TRITON, tl, triton\n\n',
                'from vllm.triton_utils import HAS_TRITON, tl, triton\n\nimport os\n\n_PDL_STATE = {"on": None}\n\n\ndef pdl_on():\n    if _PDL_STATE["on"] is None:\n        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()\n        forced = said == "interpret"\n        asked = forced or said in ("1", "true", "yes", "on")\n        on = forced or bool(\n            asked\n            and os.environ.get("TRITON_INTERPRET") != "1"\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability()[0] >= 9\n        )\n        _PDL_STATE["on"] = on\n        if asked:\n            print(f"PDL {__name__} enabled={int(on)}", flush=True)\n    return _PDL_STATE["on"]\n\n',
            ),
            (
                '    BLOCK_N: tl.constexpr,\n) -> None:\n    row = tl.program_id(0)\n',
                '    BLOCK_N: tl.constexpr,\n    PDL: tl.constexpr = False,\n) -> None:\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    row = tl.program_id(0)\n',
            ),
            (
                '    BLOCK_SPLITS: tl.constexpr,\n) -> None:\n    row = tl.program_id(0)\n',
                '    BLOCK_SPLITS: tl.constexpr,\n    PDL: tl.constexpr = False,\n) -> None:\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    row = tl.program_id(0)\n',
            ),
            (
                '    BLOCK_D: tl.constexpr,\n) -> None:\n    row = tl.program_id(0)\n',
                '    BLOCK_D: tl.constexpr,\n    PDL: tl.constexpr = False,\n) -> None:\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    row = tl.program_id(0)\n',
            ),
            (
                '    LOAD_ROPE_POSITIONS: tl.constexpr,\n) -> None:\n    row = tl.program_id(0)\n',
                '    LOAD_ROPE_POSITIONS: tl.constexpr,\n    PDL: tl.constexpr = False,\n) -> None:\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    row = tl.program_id(0)\n',
            ),
            (
                '        num_stages=2,\n    )\n',
                '        num_stages=2,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
            (
                '        num_stages=1,\n    )\n',
                '        num_stages=1,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
            (
                '            grid=(num_rows, num_kv_heads, num_splits),\n        )\n',
                '            grid=(num_rows, num_kv_heads, num_splits),\n            PDL=pdl_on(), launch_pdl=pdl_on(),\n        )\n',
            ),
            (
                '                grid=(num_rows, num_query_heads),\n            )\n',
                '                grid=(num_rows, num_query_heads),\n                PDL=pdl_on(), launch_pdl=pdl_on(),\n            )\n',
            ),
            (
                '        BLOCK_D=triton.next_power_of_2(cache.shape[3]),\n        num_warps=4,\n    )\n\n',
                '        BLOCK_D=triton.next_power_of_2(cache.shape[3]),\n        num_warps=4,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n\n',
            ),
            (
                '        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),\n        num_warps=4,\n    )\n    return pooled, first_positions\n',
                '        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),\n        num_warps=4,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n    return pooled, first_positions\n',
            ),
        ),
    ),
    'vllm/v1/attention/ops/triton_reshape_and_cache_flash.py': (
        '6cac51475b8c656992a21b2d150acd3a16a95a7ab7d49aab151a3ef13c24b80d',
        (
            (
                'from vllm.v1.kv_cache_interface import KVQuantMode\n\n',
                'from vllm.v1.kv_cache_interface import KVQuantMode\n\nimport os\n\n_PDL_STATE = {"on": None}\n\n\ndef pdl_on():\n    if _PDL_STATE["on"] is None:\n        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()\n        forced = said == "interpret"\n        asked = forced or said in ("1", "true", "yes", "on")\n        on = forced or bool(\n            asked\n            and os.environ.get("TRITON_INTERPRET") != "1"\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability()[0] >= 9\n        )\n        _PDL_STATE["on"] = on\n        if asked:\n            print(f"PDL {__name__} enabled={int(on)}", flush=True)\n    return _PDL_STATE["on"]\n\n',
            ),
            (
                '    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,\n    # FP8 flags\n    FP8_KV_CACHE: tl.constexpr,\n    # tune parameters\n    TILE_SIZE: tl.constexpr,\n):\n    token_idx = tl.program_id(axis=0)\n    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)\n    if slot_idx < 0:\n        # Padding token that should be ignored.\n        return\n',
                '    USE_HEAD_MAJOR_LAYOUT: tl.constexpr,\n    # FP8 flags\n    FP8_KV_CACHE: tl.constexpr,\n    # tune parameters\n    TILE_SIZE: tl.constexpr,\n    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    token_idx = tl.program_id(axis=0)\n    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)\n    if slot_idx < 0:\n        # Padding token that should be ignored.\n        return\n',
            ),
            (
                '    IS_INT_QUANT: tl.constexpr = False,\n):\n    tok = tl.program_id(0)\n',
                '    IS_INT_QUANT: tl.constexpr = False,\n    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    tok = tl.program_id(0)\n',
            ),
            (
                '        num_warps=num_warps,\n    )\n',
                '        num_warps=num_warps,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
            (
                '        num_warps=num_warps,\n        num_stages=num_stages,\n    )\n\n',
                '        num_warps=num_warps,\n        num_stages=num_stages,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n\n',
            ),
            (
                '):\n    token_idx = tl.program_id(axis=0)\n    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)\n    if slot_idx < 0:\n        # Padding token that should be ignored.\n        return\n\n    tile_i = tl.program_id(axis=1)\n    tile_offs = tl.arange(0, TILE_SIZE)\n\n    block_idx = slot_idx // block_size\n    block_offset = slot_idx % block_size\n\n    src_key_idx = token_idx * key_stride + tile_i * head_size_k\n    src_value_idx = token_idx * value_stride + tile_i * head_size_v\n\n    tgt_idx = (\n        block_idx * block_stride + block_offset * page_stride + tile_i * head_stride\n    )\n\n    # [TILE_SIZE]\n    key_load = tl.load(key_ptr + src_key_idx + tile_offs, mask=tile_offs < head_size_k)\n    if FP8_KV_CACHE:\n        # tl.store will do the correct implicit cast to fp8,\n        # based on the key_cache_ptr.dtype.element_ty\n        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)\n    else:\n        key_tile = key_load\n\n    # [TILE_SIZE]\n    value_load = tl.load(\n        value_ptr + src_value_idx + tile_offs, mask=tile_offs < head_size_v\n    )\n    if FP8_KV_CACHE:\n        if value_load.dtype.is_fp8():\n            value_tile = value_load\n        else:\n            # tl.store will do the correct implicit cast to fp8,\n            #  based on the value_cache_ptr.dtype.element_ty\n            value_tile = value_load / tl.load(v_scale)\n    else:\n        value_tile = value_load\n\n    tl.store(\n        kv_cache_ptr + tgt_idx + tile_offs,\n        key_tile,\n        mask=tile_offs < head_size_k,\n    )\n    tl.store(\n        kv_cache_ptr + tgt_idx + head_size_k + tile_offs,\n        value_tile,\n        mask=tile_offs < head_size_v,\n    )\n    return\n\n\ndef triton_reshape_and_cache_flash_diffkv(\n    key: torch.Tensor,  # [num_tokens, num_heads, head_size]\n    value: torch.Tensor,  # [num_tokens, num_heads, head_size_v]\n    # Strided [num_blocks, block_size, num_heads, head_size + head_size_v].\n    kv_cache: torch.Tensor,\n    slot_mapping: torch.Tensor,  # [num_tokens]\n    kv_cache_dtype: str,  # "auto", "fp8"\n    k_scale: torch.Tensor,  # float32\n    v_scale: torch.Tensor,  # float32\n):\n    num_heads = key.shape[1]\n    head_size_k = key.shape[2]\n    head_size_v = value.shape[2]\n    block_size = kv_cache.shape[1]\n\n    k_stride = key.stride()[0]\n    v_stride = value.stride()[0]\n    block_stride = kv_cache.stride()[0]\n    page_stride = kv_cache.stride()[1]\n    head_stride = kv_cache.stride()[2]\n\n    kv_cache_torch_dtype = (\n        current_platform.fp8_dtype()\n        if is_quantized_kv_cache(kv_cache_dtype)\n        else kv_cache.dtype\n    )\n\n    if kv_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(kv_cache_dtype):\n        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)\n        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)\n        kv_cache = kv_cache.view(kv_cache_torch_dtype)\n    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)\n    # heuristics instead of autotuning\n    TILE_SIZE = max(head_size_k, head_size_v)\n    TILE_SIZE = triton.next_power_of_2(TILE_SIZE)\n    if current_platform.is_rocm() or current_platform.is_xpu():\n        num_stages = 4\n        num_warps = 8\n    else:  # cuda\n        num_stages = 10\n        num_warps = 16\n\n    # TODO(ngl): maybe replace with static launch grid to avoid overhead if\n    #   using cudagraphs\n    grid = lambda meta: (\n        slot_mapping.shape[0],\n        num_heads,\n    )\n\n    reshape_and_cache_kernel_flash_diffkv[grid](\n        key_ptr=key,\n        value_ptr=value,\n        kv_cache_ptr=kv_cache,\n        slot_mapping_ptr=slot_mapping,\n        k_scale=k_scale,\n        v_scale=v_scale,\n        # strides\n        key_stride=k_stride,\n        value_stride=v_stride,\n        block_stride=block_stride,\n        page_stride=page_stride,\n        head_stride=head_stride,\n        num_heads=num_heads,\n        head_size_k=head_size_k,\n        head_size_v=head_size_v,\n        block_size=block_size,\n        # FP8 flags\n        FP8_KV_CACHE=FP8_KV_CACHE,\n        # autotune parameters\n        TILE_SIZE=TILE_SIZE,\n        num_warps=num_warps,\n        num_stages=num_stages,\n',
                '    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    token_idx = tl.program_id(axis=0)\n    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)\n    if slot_idx < 0:\n        # Padding token that should be ignored.\n        return\n\n    tile_i = tl.program_id(axis=1)\n    tile_offs = tl.arange(0, TILE_SIZE)\n\n    block_idx = slot_idx // block_size\n    block_offset = slot_idx % block_size\n\n    src_key_idx = token_idx * key_stride + tile_i * head_size_k\n    src_value_idx = token_idx * value_stride + tile_i * head_size_v\n\n    tgt_idx = (\n        block_idx * block_stride + block_offset * page_stride + tile_i * head_stride\n    )\n\n    # [TILE_SIZE]\n    key_load = tl.load(key_ptr + src_key_idx + tile_offs, mask=tile_offs < head_size_k)\n    if FP8_KV_CACHE:\n        # tl.store will do the correct implicit cast to fp8,\n        # based on the key_cache_ptr.dtype.element_ty\n        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)\n    else:\n        key_tile = key_load\n\n    # [TILE_SIZE]\n    value_load = tl.load(\n        value_ptr + src_value_idx + tile_offs, mask=tile_offs < head_size_v\n    )\n    if FP8_KV_CACHE:\n        if value_load.dtype.is_fp8():\n            value_tile = value_load\n        else:\n            # tl.store will do the correct implicit cast to fp8,\n            #  based on the value_cache_ptr.dtype.element_ty\n            value_tile = value_load / tl.load(v_scale)\n    else:\n        value_tile = value_load\n\n    tl.store(\n        kv_cache_ptr + tgt_idx + tile_offs,\n        key_tile,\n        mask=tile_offs < head_size_k,\n    )\n    tl.store(\n        kv_cache_ptr + tgt_idx + head_size_k + tile_offs,\n        value_tile,\n        mask=tile_offs < head_size_v,\n    )\n    return\n\n\ndef triton_reshape_and_cache_flash_diffkv(\n    key: torch.Tensor,  # [num_tokens, num_heads, head_size]\n    value: torch.Tensor,  # [num_tokens, num_heads, head_size_v]\n    # Strided [num_blocks, block_size, num_heads, head_size + head_size_v].\n    kv_cache: torch.Tensor,\n    slot_mapping: torch.Tensor,  # [num_tokens]\n    kv_cache_dtype: str,  # "auto", "fp8"\n    k_scale: torch.Tensor,  # float32\n    v_scale: torch.Tensor,  # float32\n):\n    num_heads = key.shape[1]\n    head_size_k = key.shape[2]\n    head_size_v = value.shape[2]\n    block_size = kv_cache.shape[1]\n\n    k_stride = key.stride()[0]\n    v_stride = value.stride()[0]\n    block_stride = kv_cache.stride()[0]\n    page_stride = kv_cache.stride()[1]\n    head_stride = kv_cache.stride()[2]\n\n    kv_cache_torch_dtype = (\n        current_platform.fp8_dtype()\n        if is_quantized_kv_cache(kv_cache_dtype)\n        else kv_cache.dtype\n    )\n\n    if kv_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(kv_cache_dtype):\n        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)\n        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)\n        kv_cache = kv_cache.view(kv_cache_torch_dtype)\n    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)\n    # heuristics instead of autotuning\n    TILE_SIZE = max(head_size_k, head_size_v)\n    TILE_SIZE = triton.next_power_of_2(TILE_SIZE)\n    if current_platform.is_rocm() or current_platform.is_xpu():\n        num_stages = 4\n        num_warps = 8\n    else:  # cuda\n        num_stages = 10\n        num_warps = 16\n\n    # TODO(ngl): maybe replace with static launch grid to avoid overhead if\n    #   using cudagraphs\n    grid = lambda meta: (\n        slot_mapping.shape[0],\n        num_heads,\n    )\n\n    reshape_and_cache_kernel_flash_diffkv[grid](\n        key_ptr=key,\n        value_ptr=value,\n        kv_cache_ptr=kv_cache,\n        slot_mapping_ptr=slot_mapping,\n        k_scale=k_scale,\n        v_scale=v_scale,\n        # strides\n        key_stride=k_stride,\n        value_stride=v_stride,\n        block_stride=block_stride,\n        page_stride=page_stride,\n        head_stride=head_stride,\n        num_heads=num_heads,\n        head_size_k=head_size_k,\n        head_size_v=head_size_v,\n        block_size=block_size,\n        # FP8 flags\n        FP8_KV_CACHE=FP8_KV_CACHE,\n        # autotune parameters\n        TILE_SIZE=TILE_SIZE,\n        num_warps=num_warps,\n        num_stages=num_stages,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n',
            ),
        ),
    ),
    'vllm/v1/attention/ops/int4_per_token_head.py': (
        'd91968f280cd8eaed7570ab255f0276b40e833e18a25066d371d9ca8149425f2',
        (
            (
                'from vllm.v1.attention.ops.triton_unified_attention import reduce_segments\n\n',
                'from vllm.v1.attention.ops.triton_unified_attention import reduce_segments\n\nimport os\n\n_PDL_STATE = {"on": None}\n\n\ndef pdl_on():\n    if _PDL_STATE["on"] is None:\n        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()\n        forced = said == "interpret"\n        asked = forced or said in ("1", "true", "yes", "on")\n        on = forced or bool(\n            asked\n            and os.environ.get("TRITON_INTERPRET") != "1"\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability()[0] >= 9\n        )\n        _PDL_STATE["on"] = on\n        if asked:\n            print(f"PDL {__name__} enabled={int(on)}", flush=True)\n    return _PDL_STATE["on"]\n\n',
            ),
            (
                '):\n    """INT4 asymmetric quantization with zero-point steganography."""\n',
                '    PDL: tl.constexpr = False,\n):\n    """INT4 asymmetric quantization with zero-point steganography."""\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n',
            ),
            (
                '        num_warps=num_warps,\n    )\n',
                '        num_warps=num_warps,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
            (
                '):\n    # Shared prologue: sequence lookup, q-block bounds, early returns.\n',
                '    PDL: tl.constexpr = False,\n):\n    # Shared prologue: sequence lookup, q-block bounds, early returns.\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n',
            ),
            (
                '        PACKING_FACTOR=packing_factor,\n    )\n',
                '        PACKING_FACTOR=packing_factor,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
            (
                '    D: tl.constexpr,\n):\n    pid = tl.program_id(0)\n',
                '    D: tl.constexpr,\n    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    pid = tl.program_id(0)\n',
            ),
            (
                '        num_warps=4,\n    )\n',
                '        num_warps=4,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
        ),
    ),
    'vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py': (
        '2c34883a6396bac0238f6308196854b653eadab8bbebbe7dd2d558e30d065cbb',
        (
            (
                'from vllm.triton_utils import tl, triton\n\n',
                'from vllm.triton_utils import tl, triton\n\nimport os\n\n_PDL_STATE = {"on": None}\n\n\ndef pdl_on():\n    if _PDL_STATE["on"] is None:\n        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()\n        forced = said == "interpret"\n        asked = forced or said in ("1", "true", "yes", "on")\n        on = forced or bool(\n            asked\n            and os.environ.get("TRITON_INTERPRET") != "1"\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability()[0] >= 9\n        )\n        _PDL_STATE["on"] = on\n        if asked:\n            print(f"PDL {__name__} enabled={int(on)}", flush=True)\n    return _PDL_STATE["on"]\n\n',
            ),
            (
                '    STAGES: tl.constexpr,\n) -> None:\n    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE\n',
                '    STAGES: tl.constexpr,\n    PDL: tl.constexpr = False,\n) -> None:\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE\n',
            ),
            (
                '    STAGES: tl.constexpr,\n) -> None:\n    num_columns = page_table_width * PAGE_SIZE\n',
                '    STAGES: tl.constexpr,\n    PDL: tl.constexpr = False,\n) -> None:\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    num_columns = page_table_width * PAGE_SIZE\n',
            ),
            (
                '    COLUMN_BLOCK: tl.constexpr,\n) -> None:\n    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1\n',
                '    COLUMN_BLOCK: tl.constexpr,\n    PDL: tl.constexpr = False,\n) -> None:\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1\n',
            ),
            (
                '            ),\n        )\n',
                '            ),\n            PDL=pdl_on(), launch_pdl=pdl_on(),\n        )\n',
            ),
            (
                '        num_warps=num_warps,\n    )\n',
                '        num_warps=num_warps,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
            (
                '        num_warps=4,\n    )\n',
                '        num_warps=4,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
            (
                '        num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,\n    )\n',
                '        num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
        ),
    ),
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            (
                'from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n\n',
                'from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n\n\n_PDL_STATE = {"on": None}\n\n\ndef pdl_on():\n    if _PDL_STATE["on"] is None:\n        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()\n        forced = said == "interpret"\n        asked = forced or said in ("1", "true", "yes", "on")\n        on = forced or bool(\n            asked\n            and os.environ.get("TRITON_INTERPRET") != "1"\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability()[0] >= 9\n        )\n        _PDL_STATE["on"] = on\n        if asked:\n            print(f"PDL {__name__} enabled={int(on)}", flush=True)\n    return _PDL_STATE["on"]\n\n',
            ),
            (
                '    BLK_HEADS: tl.constexpr,\n):\n    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n',
                '    BLK_HEADS: tl.constexpr,\n    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n',
            ),
            (
                '        num_warps=1,\n    )\n',
                '        num_warps=1,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
        ),
    ),
    'vllm/v1/worker/gpu/model_states/mamba_hybrid.py': (
        '573e130ac5587d4822fabf97adad1ad0fda02ff83c5e70df879a5bb175ae845c',
        (
            (
                'from vllm.v1.worker.utils import AttentionGroup\n\n',
                'from vllm.v1.worker.utils import AttentionGroup\n\nimport os\n\n_PDL_STATE = {"on": None}\n\n\ndef pdl_on():\n    if _PDL_STATE["on"] is None:\n        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()\n        forced = said == "interpret"\n        asked = forced or said in ("1", "true", "yes", "on")\n        on = forced or bool(\n            asked\n            and os.environ.get("TRITON_INTERPRET") != "1"\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability()[0] >= 9\n        )\n        _PDL_STATE["on"] = on\n        if asked:\n            print(f"PDL {__name__} enabled={int(on)}", flush=True)\n    return _PDL_STATE["on"]\n\n',
            ),
            (
                '                    self.num_accepted_tokens_gpu,\n                )\n',
                '                    self.num_accepted_tokens_gpu,\n                    PDL=pdl_on(), launch_pdl=pdl_on(),\n                )\n',
            ),
            (
                '                    max(num_sampled, 1),\n                )\n',
                '                    max(num_sampled, 1),\n                    PDL=pdl_on(), launch_pdl=pdl_on(),\n                )\n',
            ),
            (
                '    num_accepted_ptr,  # [max_num_reqs]\n):\n    row = tl.program_id(0)\n',
                '    num_accepted_ptr,  # [max_num_reqs]\n    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    row = tl.program_id(0)\n',
            ),
            (
                '    num_sampled,\n):\n    row = tl.program_id(0)\n',
                '    num_sampled,\n    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    row = tl.program_id(0)\n',
            ),
        ),
    ),
    'vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py': (
        '000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f',
        (
            (
                'from vllm.triton_utils import tl, triton\n\n',
                'from vllm.triton_utils import tl, triton\n\nimport os\n\n_PDL_STATE = {"on": None}\n\n\ndef pdl_on():\n    if _PDL_STATE["on"] is None:\n        said = str(os.environ.get("TAAF_VLLM_PDL") or "").strip().lower()\n        forced = said == "interpret"\n        asked = forced or said in ("1", "true", "yes", "on")\n        on = forced or bool(\n            asked\n            and os.environ.get("TRITON_INTERPRET") != "1"\n            and torch.cuda.is_available()\n            and torch.cuda.get_device_capability()[0] >= 9\n        )\n        _PDL_STATE["on"] = on\n        if asked:\n            print(f"PDL {__name__} enabled={int(on)}", flush=True)\n    return _PDL_STATE["on"]\n\n',
            ),
            (
                '    IS_KDA: tl.constexpr,\n):\n    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n',
                '    IS_KDA: tl.constexpr,\n    PDL: tl.constexpr = False,\n):\n    if PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n',
            ),
            (
                '        num_stages=num_stages,\n    )\n',
                '        num_stages=num_stages,\n        PDL=pdl_on(), launch_pdl=pdl_on(),\n    )\n',
            ),
        ),
    ),
}
