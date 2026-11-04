TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            ('        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n        core_attn_out = torch.zeros(\n            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n            dtype=hidden_states.dtype,\n            device=hidden_states.device,\n        )\n',
             '        # see discussions in https://github.com/vllm-project/vllm/pull/28182\n        from vllm import taaf_gdn_mixed_full as taaf_tail\n\n        core_attn_out = (torch.empty if taaf_tail.tail_owned(self.prefix) else torch.zeros)(\n            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),\n            dtype=hidden_states.dtype,\n            device=hidden_states.device,\n        )\n'),
        ),
    ),
}
