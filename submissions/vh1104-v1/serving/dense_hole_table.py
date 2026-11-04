TARGETS = {
    'vllm/model_executor/layers/linear.py': (
        '094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d',
        (
            (
                ('                layer.weight.data = weight.t().contiguous().t()\n'
                 '\n'
                 '    def apply(\n'
                 '        self,\n'
                 '        layer: torch.nn.Module,\n'
                 '        x: torch.Tensor,\n'
                 '        bias: torch.Tensor | None = None,\n'
                 '    ) -> torch.Tensor:\n'
                 '        if envs.VLLM_BATCH_INVARIANT and (\n'),
                ('                layer.weight.data = weight.t().contiguous().t()\n'
                 '        from vllm import taaf_dense_hole\n'
                 '\n'
                 '        taaf_dense_hole.after_loading(layer)\n'
                 '\n'
                 '    def apply(\n'
                 '        self,\n'
                 '        layer: torch.nn.Module,\n'
                 '        x: torch.Tensor,\n'
                 '        bias: torch.Tensor | None = None,\n'
                 '    ) -> torch.Tensor:\n'
                 '        if bias is None and getattr(layer, "taaf_dense_hole", False):\n'
                 '            from vllm import taaf_dense_hole\n'
                 '\n'
                 '            return taaf_dense_hole.apply(layer, x)\n'
                 '        if envs.VLLM_BATCH_INVARIANT and (\n'),
            ),
        ),
    ),
}
