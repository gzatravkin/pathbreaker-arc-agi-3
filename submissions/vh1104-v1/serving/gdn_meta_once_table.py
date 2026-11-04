TARGETS = {
    'vllm/v1/attention/backends/gdn_attn.py': (
        '3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054',
        (
            (
                ('from vllm.utils.torch_utils import async_tensor_h2d\n'),
                ('from vllm import taaf_gdn_meta_once\n'
                 'from vllm.utils.torch_utils import async_tensor_h2d\n'),
            ),
            (
                ('        self.num_accepted_tokens: torch.Tensor = torch.empty(\n'
                 '            (self.decode_cudagraph_max_bs,),\n'
                 '            dtype=torch.int32,\n'
                 '            device=device,\n'
                 '        )\n'),
                ('        self.num_accepted_tokens: torch.Tensor = torch.empty(\n'
                 '            (self.decode_cudagraph_max_bs,),\n'
                 '            dtype=torch.int32,\n'
                 '            device=device,\n'
                 '        )\n'
                 '        self.taaf_meta_mask = None\n'
                 '        taaf_gdn_meta_once.meta_share(self)\n'),
            ),
            (
                ('        m = common_attn_metadata\n'
                 '\n'
                 '        query_start_loc = m.query_start_loc\n'),
                ('        m = common_attn_metadata\n'
                 '        taaf_meta_key = taaf_gdn_meta_once.meta_key(\n'
                 '            self, m, num_accepted_tokens, num_decode_draft_tokens_cpu\n'
                 '        )\n'
                 '        if taaf_meta_key is not None:\n'
                 '            taaf_meta_made = taaf_gdn_meta_once.meta_derive(\n'
                 '                self,\n'
                 '                m,\n'
                 '                taaf_meta_key,\n'
                 '                mamba_get_block_table_tensor,\n'
                 '                NULL_BLOCK_ID,\n'
                 '                taaf_gdn_accept.commit_for,\n'
                 '            )\n'
                 '            if taaf_meta_made is None:\n'
                 '                taaf_meta_made = taaf_gdn_meta_once.meta_base(\n'
                 '                    self,\n'
                 '                    taaf_meta_key,\n'
                 '                    lambda: self.build(\n'
                 '                        common_prefix_len,\n'
                 '                        m,\n'
                 '                        num_accepted_tokens,\n'
                 '                        num_decode_draft_tokens_cpu,\n'
                 '                        fast_build,\n'
                 '                    ),\n'
                 '                )\n'
                 '            return taaf_meta_made\n'
                 '\n'
                 '        query_start_loc = m.query_start_loc\n'),
            ),
            (
                ('        if spec_sequence_masks is None:\n'
                 '            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n'),
                ('        self.taaf_meta_mask = spec_sequence_masks_cpu\n'
                 '        if spec_sequence_masks is None:\n'
                 '            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/attn_utils.py': (
        'b9b81e59dda2720b1f9c434471381f879228832443e52756e469b4ad4905888f',
        (
            (
                ('    attn_metadata: dict[str, Any] = {}\n'
                 '    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)\n'),
                ('    from vllm import taaf_gdn_meta_once as taaf_gdn_meta\n'
                 '\n'
                 '    taaf_gdn_meta.meta_open()\n'
                 '    attn_metadata: dict[str, Any] = {}\n'
                 '    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)\n'),
            ),
            (
                ('            for layer_name in attn_group.layer_names:\n'
                 '                attn_metadata[layer_name] = metadata\n'
                 '    return attn_metadata\n'),
                ('            for layer_name in attn_group.layer_names:\n'
                 '                attn_metadata[layer_name] = metadata\n'
                 '    taaf_gdn_meta.meta_close()\n'
                 '    return attn_metadata\n'),
            ),
        ),
    ),
}
