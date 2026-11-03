TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        None,
        (
            ('        if use_fused_gdn_decode:\n            core_attn_out = torch.zeros(\n',
             '        if use_fused_gdn_decode:\n            from vllm import taaf_gdn_fused as taaf_decode_tail\n\n'
             '            core_attn_out = (torch.empty if taaf_decode_tail.decode_tail_owned(self, num_tokens)\n'
             '                             else torch.zeros)(\n'),
        ),
    ),
}
