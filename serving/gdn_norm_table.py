TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            (
                ('        assert out.shape == x_shape\n'
                 '        x_2d = x.reshape(-1, x_shape[-1])\n'),
                ('        assert out.shape == x_shape\n'
                 '        from vllm import taaf_gdn_norm\n'
                 '\n'
                 '        if taaf_gdn_norm.gated(self, x, output_gate, out):\n'
                 '            return\n'
                 '        x_2d = x.reshape(-1, x_shape[-1])\n'),
            ),
        ),
    ),
}
