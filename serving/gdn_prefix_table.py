TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            (
                ('from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n'),
                ('from vllm import taaf_gdn_prefix\n'
                 'from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n'),
            ),
            (
                ('        num_actual_tokens = attn_metadata.num_actual_tokens\n'
                 '        num_accepted_tokens = attn_metadata.num_accepted_tokens\n'
                 '\n'
                 '        mixed_qkv = mixed_qkv[:num_actual_tokens]\n'),
                ('        num_actual_tokens = attn_metadata.num_actual_tokens\n'
                 '        num_accepted_tokens = attn_metadata.num_accepted_tokens\n'
                 '        taaf_prefix = taaf_gdn_prefix.prefix(attn_metadata.spec_token_indx)\n'
                 '\n'
                 '        mixed_qkv = mixed_qkv[:num_actual_tokens]\n'),
            ),
            (
                ('            else:\n'
                 '                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)\n'
                 '                a_spec = a.index_select(0, spec_token_indx)\n'
                 '                b_spec = b.index_select(0, spec_token_indx)\n'
                 '                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)\n'),
                ('            elif taaf_prefix is not None:\n'
                 '                mixed_qkv_spec = mixed_qkv[:taaf_prefix]\n'
                 '                a_spec = a[:taaf_prefix]\n'
                 '                b_spec = b[:taaf_prefix]\n'
                 '                mixed_qkv_non_spec = mixed_qkv[taaf_prefix:]\n'
                 '            else:\n'
                 '                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)\n'
                 '                a_spec = a.index_select(0, spec_token_indx)\n'
                 '                b_spec = b.index_select(0, spec_token_indx)\n'
                 '                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)\n'),
            ),
            (
                ('            if spec_sequence_masks is not None:\n'
                 '                a_non_spec = a.index_select(0, non_spec_token_indx)\n'
                 '                b_non_spec = b.index_select(0, non_spec_token_indx)\n'),
                ('            if taaf_prefix is not None:\n'
                 '                a_non_spec = a[taaf_prefix:]\n'
                 '                b_non_spec = b[taaf_prefix:]\n'
                 '            elif spec_sequence_masks is not None:\n'
                 '                a_non_spec = a.index_select(0, non_spec_token_indx)\n'
                 '                b_non_spec = b.index_select(0, non_spec_token_indx)\n'),
            ),
            (
                ('        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:\n'
                 '            merged_out = torch.empty(\n'),
                ('        if taaf_prefix is not None and core_attn_out_non_spec is not None:\n'
                 '            core_attn_out[:taaf_prefix] = core_attn_out_spec.squeeze(0)\n'
                 '            core_attn_out[taaf_prefix:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)\n'
                 '        elif spec_sequence_masks is not None and core_attn_out_non_spec is not None:\n'
                 '            merged_out = torch.empty(\n'),
            ),
        ),
    ),
    'vllm/v1/attention/backends/gdn_attn.py': (
        '3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054',
        (
            (
                ('                non_spec_token_indx = index[:num_non_spec_tokens]\n'
                 '                spec_token_indx = index[num_non_spec_tokens:]\n'),
                ('                non_spec_token_indx = index[:num_non_spec_tokens]\n'
                 '                spec_token_indx = index[num_non_spec_tokens:]\n'
                 '                from vllm import taaf_gdn_prefix\n'
                 '\n'
                 '                taaf_gdn_prefix.mark(\n'
                 '                    spec_token_indx,\n'
                 '                    spec_sequence_masks_cpu,\n'
                 '                    query_start_loc_cpu,\n'
                 '                    num_spec_decode_tokens,\n'
                 '                )\n'),
            ),
        ),
    ),
}
