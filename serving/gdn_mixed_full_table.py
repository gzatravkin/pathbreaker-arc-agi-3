TARGETS = {
    'vllm/v1/attention/backends/gdn_attn.py': (
        '3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054',
        (
            (
                ('    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH\n'),
                ('    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH\n'
                 '\n'
                 '    @classmethod\n'
                 '    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):\n'
                 '        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        return taaf_gdn_mixed_full.support(cls._cudagraph_support, AttentionCGSupport.ALWAYS)\n'),
            ),
            (
                ('        query_start_loc_cpu = m.query_start_loc_cpu\n'
                 '        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None\n'),
                ('        query_start_loc_cpu = m.query_start_loc_cpu\n'
                 '        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        taaf_mixed_made = taaf_gdn_mixed_full.build(\n'
                 '            self,\n'
                 '            m,\n'
                 '            num_accepted_tokens,\n'
                 '            num_decode_draft_tokens_cpu,\n'
                 '            globals().get("GDNAcceptMetadata")\n'
                 '            if getattr(self, "gdn_accept", False)\n'
                 '            else GDNAttentionMetadata,\n'
                 '            mamba_get_block_table_tensor,\n'
                 '            NULL_BLOCK_ID,\n'
                 '            getattr(globals().get("taaf_gdn_accept"), "commit_for", None),\n'
                 '        )\n'
                 '        if taaf_mixed_made is not None:\n'
                 '            return taaf_mixed_made\n'
                 '        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None\n'),
            ),
            (
                ('        This method builds the metadata for full cudagraph capture.\n'
                 '        Currently, only decode is supported for full cudagraphs with Mamba.\n'
                 '        """\n'
                 '        m = common_attn_metadata\n'),
                ('        This method builds the metadata for full cudagraph capture.\n'
                 '        Currently, only decode is supported for full cudagraphs with Mamba.\n'
                 '        """\n'
                 '        m = common_attn_metadata\n'
                 '        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        taaf_mixed_made = taaf_gdn_mixed_full.capture(\n'
                 '            self,\n'
                 '            m,\n'
                 '            globals().get("GDNAcceptMetadata")\n'
                 '            if getattr(self, "gdn_accept", False)\n'
                 '            else GDNAttentionMetadata,\n'
                 '            mamba_get_block_table_tensor,\n'
                 '            NULL_BLOCK_ID,\n'
                 '            getattr(globals().get("taaf_gdn_accept"), "commit_for", None),\n'
                 '        )\n'
                 '        if taaf_mixed_made is not None:\n'
                 '            return taaf_mixed_made\n'),
            ),
        ),
    ),
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            (
                ('        assert isinstance(attn_metadata, GDNAttentionMetadata)\n'
                 '\n'
                 '        if (\n'
                 '            self.enable_packed_recurrent_decode\n'),
                ('        assert isinstance(attn_metadata, GDNAttentionMetadata)\n'
                 '\n'
                 '        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        if taaf_gdn_mixed_full.owned(attn_metadata) is not None:\n'
                 '            return taaf_gdn_mixed_full.forward(\n'
                 '                self, mixed_qkv, b, a, core_attn_out, attn_metadata\n'
                 '            )\n'
                 '\n'
                 '        if (\n'
                 '            self.enable_packed_recurrent_decode\n'),
            ),
        ),
    ),
    'vllm/v1/attention/backends/short_conv_attn.py': (
        '84fc460ef2fbd09743b069bec0c14b39542af6c2254a403e37a048427e151121',
        (
            (
                ('    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH\n'
                 '    reorder_batch_threshold: int = 1\n'
                 '    supports_update_block_table = False\n'),
                ('    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH\n'
                 '    reorder_batch_threshold: int = 1\n'
                 '    supports_update_block_table = False\n'
                 '\n'
                 '    @classmethod\n'
                 '    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):\n'
                 '        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        return taaf_gdn_mixed_full.short_conv_support(cls._cudagraph_support, AttentionCGSupport.ALWAYS)\n'),
            ),
            (
                ('        m = common_attn_metadata\n'
                 '        spec_sequence_masks_cpu: torch.Tensor | None = None\n'),
                ('        m = common_attn_metadata\n'
                 '        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        taaf_mixed_made = taaf_gdn_mixed_full.build_ple(\n'
                 '            self,\n'
                 '            m,\n'
                 '            num_accepted_tokens,\n'
                 '            num_decode_draft_tokens_cpu,\n'
                 '            PleShortConvAttentionMetadata,\n'
                 '            mamba_get_block_table_tensor,\n'
                 '            NULL_BLOCK_ID,\n'
                 '        )\n'
                 '        if taaf_mixed_made is not None:\n'
                 '            return taaf_mixed_made\n'
                 '        spec_sequence_masks_cpu: torch.Tensor | None = None\n'),
            ),
            (
                ('        m = common_attn_metadata\n'
                 '        assert (\n'
                 '            m.num_reqs <= self.decode_cudagraph_max_bs\n'
                 '            and m.num_actual_tokens <= self.decode_cudagraph_max_tokens\n'),
                ('        m = common_attn_metadata\n'
                 '        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        taaf_mixed_made = taaf_gdn_mixed_full.capture_ple(\n'
                 '            self, m, PleShortConvAttentionMetadata, mamba_get_block_table_tensor, NULL_BLOCK_ID\n'
                 '        )\n'
                 '        if taaf_mixed_made is not None:\n'
                 '            return taaf_mixed_made\n'
                 '        assert (\n'
                 '            m.num_reqs <= self.decode_cudagraph_max_bs\n'
                 '            and m.num_actual_tokens <= self.decode_cudagraph_max_tokens\n'),
            ),
        ),
    ),
    'vllm/models/qwen4_exp/nvidia/ple_layer.py': (
        'b81f9372eca168157c7f3cf5646d61eb0f139b64c9420588d8288672f18a3fd2',
        (
            (
                ('        num_prefills = metadata.num_prefills\n'
                 '        num_decodes = metadata.num_decodes\n'
                 '        num_decode_tokens = metadata.num_decode_tokens\n'),
                ('        from vllm import taaf_gdn_mixed_full\n'
                 '\n'
                 '        if taaf_gdn_mixed_full.ple_owned(metadata) is not None:\n'
                 '            return taaf_gdn_mixed_full.ple_forward(\n'
                 '                self,\n'
                 '                inputs,\n'
                 '                residual,\n'
                 '                outer_residual,\n'
                 '                metadata,\n'
                 '                conv_state,\n'
                 '                conv_weights,\n'
                 '            )\n'
                 '        num_prefills = metadata.num_prefills\n'
                 '        num_decodes = metadata.num_decodes\n'
                 '        num_decode_tokens = metadata.num_decode_tokens\n'),
            ),
        ),
    ),
}
