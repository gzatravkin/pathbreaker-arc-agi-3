TARGETS = {
    'vllm/model_executor/layers/quantization/auto_gptq.py': (
        '3efa1ab7ee3648a1239dcb2f7e0f9528a8229e7b8e36ba1ff6ac300b58c1d226',
        (
            (
                ('    def process_weights_after_loading(self, layer: RoutedExperts) -> None:\n'
                 '        def replace_or_register(name: str, val: torch.Tensor | None):\n'),
                ('    def process_weights_after_loading(self, layer: RoutedExperts) -> None:\n'
                 '        from vllm import taaf_humming_int3\n'
                 '\n'
                 '        if taaf_humming_int3.wanted(self, layer):\n'
                 '            taaf_humming_int3.convert(layer, self)\n'
                 '            self._setup_kernel(layer)\n'
                 '            return\n'
                 '\n'
                 '        def replace_or_register(name: str, val: torch.Tensor | None):\n'),
            ),
        ),
    ),
}
