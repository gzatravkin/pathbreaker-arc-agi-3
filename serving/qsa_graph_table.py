TARGETS = {
    'vllm/models/qwen4_exp/common/qsa_cache.py': (
        'c6651a0cf27e2ffb23b90eafe21e044df5609ce976f52ebffa7a4fbea688fc04',
        (
            (
                '    compress_ratio: int\n'
                '\n'
                '\n'
                'class QSAMetadataBuilder(',
                '    compress_ratio: int\n'
                '    bucket: object = None\n'
                '\n'
                '\n'
                'class QSAMetadataBuilder(',
            ),
            (
                '    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n'
                '\n'
                '    def __init__(\n'
                '        self,\n'
                '        kv_cache_spec: AttentionSpec,',
                '    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n'
                '\n'
                '    @classmethod\n'
                '    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):\n'
                '        from vllm.taaf_qsa_graph import declared_support\n'
                '\n'
                '        return declared_support(cls._cudagraph_support)\n'
                '\n'
                '    def __init__(\n'
                '        self,\n'
                '        kv_cache_spec: AttentionSpec,',
            ),
            (
                '        return QSAForwardMetadata(\n'
                '            block_table=common_attn_metadata.block_table_tensor,',
                '        from vllm.taaf_qsa_graph import prepare\n'
                '\n'
                '        bucket = prepare(\n'
                '            self, common_attn_metadata, num_decodes, num_decode_tokens, visible_blocks\n'
                '        )\n'
                '        return QSAForwardMetadata(\n'
                '            block_table=common_attn_metadata.block_table_tensor,',
            ),
            (
                '            compress_ratio=self.compress_ratio,\n'
                '        )\n'
                '\n'
                '\n'
                'class QSAStateBackend(',
                '            compress_ratio=self.compress_ratio,\n'
                '            bucket=bucket,\n'
                '        )\n'
                '\n'
                '\n'
                'class QSAStateBackend(',
            ),
        ),
    ),
    'vllm/models/qwen4_exp/nvidia/indexer_qsa.py': (
        '23a39cc4ee463bd3441e4343d9a0933ab1a81ac5a3f5a5af80f5b084f70d8fef',
        (
            (
                '        # Decode requests occupy the leading rows and share one query length.\n'
                '        if num_decode_tokens:\n',
                '        bucket = compressed_metadata.bucket\n'
                '        if bucket is not None:\n'
                '            from vllm.taaf_qsa_graph import bucket_select\n'
                '\n'
                '            bucket_select(\n'
                '                qsa_select_paged_decode,\n'
                '                qsa_select_paged_prefill,\n'
                '                q,\n'
                '                compressed_key_cache,\n'
                '                compressed_metadata,\n'
                '                self.token_topk,\n'
                '                self.compress_ratio,\n'
                '                block_indices,\n'
                '            )\n'
                '\n'
                '        # Decode requests occupy the leading rows and share one query length.\n'
                '        if bucket is None and num_decode_tokens:\n',
            ),
            (
                '        if num_decode_tokens < num_tokens:\n'
                '            num_decodes = compressed_metadata.num_decodes\n'
                '            prefill_slice',
                '        if bucket is None and num_decode_tokens < num_tokens:\n'
                '            num_decodes = compressed_metadata.num_decodes\n'
                '            prefill_slice',
            ),
        ),
    ),
    'vllm/models/qwen4_exp/nvidia/qsa.py': (
        '9ac72b0e5f02341704a2f5a49bf2f4a6f141d2e241e9db07ea5918592bac90ea',
        (
            (
                '    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n'
                '\n'
                '\n'
                'class Qwen4ExpQSAFlashAttentionBackend(',
                '    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n'
                '\n'
                '    @classmethod\n'
                '    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):\n'
                '        from vllm.taaf_qsa_graph import declared_support\n'
                '\n'
                '        return declared_support(cls._cudagraph_support)\n'
                '\n'
                '\n'
                'class Qwen4ExpQSAFlashAttentionBackend(',
            ),
            (
                '            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,\n',
                '            use_prefill_config=__import__(\n'
                '                "vllm.taaf_qsa_graph", fromlist=["prefill_flag"]\n'
                '            ).prefill_flag(\n'
                '                side_metadata.bucket,\n'
                '                main_metadata.max_query_len > self._max_decode_query_len,\n'
                '            ),\n',
            ),
        ),
    ),
}
