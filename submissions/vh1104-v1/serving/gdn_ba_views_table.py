TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            ('        torch.ops.vllm.qwen_gdn_attention_core(\n            mixed_qkv,\n            b.contiguous(),\n            a.contiguous(),\n            core_attn_out,\n            layer_name=_encode_layer_name(self.prefix),\n        )\n',
             '        torch.ops.vllm.qwen_gdn_attention_core(\n            mixed_qkv,\n            b,\n            a,\n            core_attn_out,\n            layer_name=_encode_layer_name(self.prefix),\n        )\n'),
            ('            core_attn_out: Pre-allocated output buffer for attention results.\n        """\n        forward_context = get_forward_context()\n        attn_metadata_raw = forward_context.attn_metadata\n\n        attn_metadata = None\n        if isinstance(attn_metadata_raw, dict):\n            attn_metadata = attn_metadata_raw.get(self.prefix)\n        if attn_metadata is None:\n            self._warmup_prefill_kernels(mixed_qkv, 0)\n            return\n',
             '            core_attn_out: Pre-allocated output buffer for attention results.\n        """\n        forward_context = get_forward_context()\n        attn_metadata_raw = forward_context.attn_metadata\n\n        attn_metadata = None\n        if isinstance(attn_metadata_raw, dict):\n            attn_metadata = attn_metadata_raw.get(self.prefix)\n        if attn_metadata is None:\n            self._warmup_prefill_kernels(mixed_qkv, 0)\n            return\n\n        from vllm import taaf_gdn_mixed_full as taaf_ba_views\n\n        if taaf_ba_views.owned(attn_metadata) is None:\n            b = b.contiguous()\n            a = a.contiguous()\n'),
        ),
    ),
}
