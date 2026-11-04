TARGETS = {
    'vllm/v1/worker/gpu/mm/encoder_cache.py': (
        '89c500338fddbeee7c4fca978a3a7c4c566a03e848e399d303fbee3c80631cd2',
        (
            (
                ('    def free_encoder_cache(self, mm_hash: str) -> None:\n'
                 '        self.encoder_outputs.pop(mm_hash, None)'),
                ('    def free_encoder_cache(self, mm_hash: str) -> None:\n'
                 '        from vllm import taaf_vision_cache\n'
                 '\n'
                 '        taaf_vision_cache.release(self, mm_hash)'),
            ),
            (
                ('        self.encoder_outputs.clear()\n'),
                ('        self.encoder_outputs.clear()\n'
                 '        from vllm import taaf_vision_cache\n'
                 '\n'
                 '        taaf_vision_cache.clear()\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/mm/encoder_runner.py': (
        '0981e352d70301b7154a217a48665ffe35f215d053ec5740c2515a3e1e11b4e6',
        (
            (
                ('                if mm_feature.identifier in self.encoder_cache.encoder_outputs:\n'
                 '                    continue\n'),
                ('                if mm_feature.identifier in self.encoder_cache.encoder_outputs:\n'
                 '                    continue\n'
                 '                from vllm import taaf_vision_cache\n'
                 '\n'
                 '                if taaf_vision_cache.restore(self.encoder_cache, mm_feature.identifier):\n'
                 '                    continue\n'),
            ),
        ),
    ),
}
