TARGETS = {
    "vllm/models/qwen4_exp/nvidia/qsa.py": (
        None,
        (
            ('    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:\n'
             '        return FullAttentionSpec(\n'
             '            block_size=vllm_config.cache_config.block_size,\n'
             '            num_kv_heads=self.num_kv_heads,\n'
             '            head_size=self.head_dim,\n'
             '            head_size_v=self.head_dim,\n'
             '            dtype=self.kv_cache_torch_dtype,\n'
             '            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n'
             '        )\n',
             '    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:\n'
             '        from vllm.taaf_qsa_hisparse import main_spec\n'
             '\n'
             '        return main_spec(self, vllm_config, FullAttentionSpec(\n'
             '            block_size=vllm_config.cache_config.block_size,\n'
             '            num_kv_heads=self.num_kv_heads,\n'
             '            head_size=self.head_dim,\n'
             '            head_size_v=self.head_dim,\n'
             '            dtype=self.kv_cache_torch_dtype,\n'
             '            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n'
             '        ))\n'),
            ('        static_context[self.layer_name] = self\n'
             '\n'
             '    def get_attn_backend(self) -> type[AttentionBackend]:\n',
             '        static_context[self.layer_name] = self\n'
             '        from vllm.taaf_qsa_hisparse import attach\n'
             '\n'
             '        attach(self, vllm_config)\n'
             '\n'
             '    def get_attn_backend(self) -> type[AttentionBackend]:\n'),
            ('        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)\n'
             '        impl.do_kv_cache_update(\n',
             '        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)\n'
             '        if self.hisparse_cache is not None:\n'
             '            from vllm.taaf_qsa_hisparse import run\n'
             '\n'
             '            run(\n'
             '                self,\n'
             '                impl,\n'
             '                query,\n'
             '                key,\n'
             '                value,\n'
             '                main_metadata,\n'
             '                side_metadata,\n'
             '                output,\n'
             '                output_gate,\n'
             '                main_metadata.max_query_len > self._max_decode_query_len,\n'
             '            )\n'
             '            return\n'
             '        impl.do_kv_cache_update(\n'),
        ),
    ),
    "vllm/models/qwen4_exp/common/qsa_cache.py": (
        None,
        (
            ('        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)\n',
             '        from vllm.taaf_qsa_hisparse import ring_capacity\n'
             '\n'
             '        capacity = ring_capacity(self.compress_ratio, span, self.cache_config.block_size)\n'),
            ('        del vllm_config\n'
             '        return MLAAttentionSpec(\n'
             '            block_size=self.cache_config.block_size,\n'
             '            num_kv_heads=1,\n'
             '            head_size=self.head_size,\n'
             '            dtype=self.dtype,\n'
             '            tokens_per_state=self.compress_ratio,\n'
             '        )\n',
             '        from vllm.taaf_qsa_hisparse import indexer_spec\n'
             '\n'
             '        return indexer_spec(vllm_config, MLAAttentionSpec(\n'
             '            block_size=self.cache_config.block_size,\n'
             '            num_kv_heads=1,\n'
             '            head_size=self.head_size,\n'
             '            dtype=self.dtype,\n'
             '            tokens_per_state=self.compress_ratio,\n'
             '        ))\n'),
        ),
    ),
    "vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py": (
        None,
        (
            ('    kv_cache = owner.kv_cache\n'
             '    assert kv_cache.numel()\n',
             '    from vllm.taaf_qsa_hisparse import warm_cache\n'
             '\n'
             '    kv_cache = warm_cache(owner)\n'
             '    assert kv_cache.numel()\n'),
        ),
    ),
    "vllm/models/qwen4_exp/config.py": (
        "19201904337294108edc713e3bc211a995273a9c872628348f4e2f0cce552f90",
        (
            ('                "QSA indexer_head_dim must cover the attention rotary "\n'
             '                f"dimension, got {values[\'indexer_head_dim\']} < {rotary_dim}"\n'
             '            )\n',
             '                "QSA indexer_head_dim must cover the attention rotary "\n'
             '                f"dimension, got {values[\'indexer_head_dim\']} < {rotary_dim}"\n'
             '            )\n'
             '        from vllm.taaf_qsa_hisparse import index_topk\n'
             '\n'
             '        self.index_topk = index_topk(\n'
             '            values["indexer_budget"], values["indexer_compress_ratio"]\n'
             '        )\n'),
            ('        self.image_token_id = image_token_id\n',
             '        if hasattr(self.text_config, "index_topk"):\n'
             '            self.index_topk = self.text_config.index_topk\n'
             '        self.image_token_id = image_token_id\n'),
        ),
    ),
    "vllm/v1/hisparse/layout.py": (
        "7ed13573e57eb9b9dec3a9a8901a080501a6b71dc3d5c25d1be1a73df11a01cd",
        (
            ('    current: list[tuple[str, MLAAttentionSpec]] = []\n'
             '    current_page = 0\n'
             '    for unit in hot_units:\n'
             '        unit_page = sum(spec.page_size_bytes for _, spec in unit)\n'
             '        if current and current_page + unit_page > indexer_page:\n',
             '    from vllm.taaf_qsa_hisparse import hot_pack_limit\n'
             '\n'
             '    pack_limit = hot_pack_limit(groups[1:], indexer_page)\n'
             '    current: list[tuple[str, MLAAttentionSpec]] = []\n'
             '    current_page = 0\n'
             '    for unit in hot_units:\n'
             '        unit_page = sum(spec.page_size_bytes for _, spec in unit)\n'
             '        if current and current_page + unit_page > cdiv(pack_limit, unit_page) * unit_page:\n'),
            ('    mla_group_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)\n'
             '    assert mla_group_spec is not None\n',
             '    from vllm.taaf_qsa_hisparse import mla_group_spec as _taaf_mla_group_spec\n'
             '\n'
             '    mla_group_spec = _taaf_mla_group_spec(UniformTypeKVCacheSpecs, mla_specs)\n'
             '    assert mla_group_spec is not None\n'),
            ('    block_sizes = {\n'
             '        spec.block_size for spec in (*source_specs.values(), *indexer_specs.values())\n'
             '    }\n'
             '    if len(block_sizes) != 1:\n'
             '        raise ValueError("HiSparse requires one resolved GPU block size.")\n'
             '    gpu_block_size = block_sizes.pop()\n',
             '    from vllm.taaf_qsa_hisparse import gpu_block_size as _taaf_gpu_block_size\n'
             '\n'
             '    gpu_block_size = _taaf_gpu_block_size(source_specs, indexer_specs)\n'),
        ),
    ),
    "vllm/v1/hisparse/binding.py": (
        "00b43de98f161e035c088786335cb73c0079ebd21d67c689ea4997f7bc0af542",
        (
            ('    block_sizes = {spec.block_size for spec in mla_specs.values()}\n'
             '    if len(block_sizes) != 1:\n'
             '        raise ValueError("HiSparse requires one scheduler block size.")\n'
             '    backends = [attn_layers[name].get_attn_backend() for name in mla_specs]\n'
             '    try:\n'
             '        block_size = select_common_block_size(block_sizes.pop(), backends)\n'
             '    except ValueError as error:\n'
             '        raise ValueError(\n'
             '            "HiSparse requires a GPU block size supported by every sparse "\n'
             '            f"attention and indexer backend: {error}"\n'
             '        ) from error\n'
             '    kv_cache_spec.update(\n'
             '        (name, spec.copy_with_new_block_size(block_size))\n'
             '        for name, spec in mla_specs.items()\n'
             '    )\n',
             '    from vllm.taaf_qsa_hisparse import resolved_block_sizes\n'
             '\n'
             '    kv_cache_spec.update(\n'
             '        resolved_block_sizes(\n'
             '            mla_specs,\n'
             '            lambda name: attn_layers[name].get_attn_backend(),\n'
             '            select_common_block_size,\n'
             '        )\n'
             '    )\n'),
        ),
    ),
    "vllm/v1/hisparse/coordinator.py": (
        "d46e0c6357a0b22147ce7a98d2bc4fee0f0a19a6143982a910ffae91195a4471",
        (
            ('            if self.gpu_pool.get_num_free_blocks() >= self.transition_watermark:\n'
             '                return\n',
             '            if self.gpu_pool.get_num_free_blocks() >= self.transition_watermark:\n'
             '                from vllm.taaf_qsa_hisparse import early_hot\n'
             '\n'
             '                if not early_hot(self, request_id):\n'
             '                    return\n'),
            ('        self._owners: dict[int, set[tuple[str, int]]] = {}\n',
             '        self._owners: dict[int, set[tuple[str, int]]] = {}\n'
             '        from vllm.taaf_qsa_hisparse import announce_admit, arm_hot_release\n'
             '\n'
             '        announce_admit(self, kv_cache_config)\n'
             '        arm_hot_release(self)\n'),
        ),
    ),
    "vllm/v1/hisparse/runtime.py": (
        "84a5305c3aa4a047c8ecb2906f6354b0d8bfe3dcd40e34f5a7422ddc15f56344",
        (
            ('        min_device_buffer_size = max_query_len * model_top_k\n',
             '        from vllm.taaf_qsa_hisparse import hot_floor_queries\n'
             '\n'
             '        min_device_buffer_size = hot_floor_queries(max_query_len) * model_top_k\n'),
            ('        self.submit_layer_mirror: Callable[[], None] | None = None\n',
             '        self.submit_layer_mirror: Callable[[], None] | None = None\n'
             '        self.graph_mirror = False\n'),
            ('            and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL\n',
             '            and (self.graph_mirror or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL)\n'),
        ),
    ),
}
