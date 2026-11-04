TARGETS = {
    'vllm/multimodal/registry.py': (
        '265dfe93a97ee9a25851d21d59a22fa7364d126324842b0c669fe598117081bb',
        (
            (
                ('        if not is_ipc_supported:\n'
                 '            return "processor_only"\n'),
                ('        from vllm import taaf_mm_processor_only\n'
                 '\n'
                 '        if not is_ipc_supported or taaf_mm_processor_only.on():\n'
                 '            return "processor_only"\n'),
            ),
        ),
    ),
    'vllm/multimodal/cache.py': (
        '4e115f755e8b90817e8d5b94d729daf32acd6c13ce06db7590bc42f191c4b7ea',
        (
            (
                ('        self.cache_if_fits(self._cache, mm_hash, MultiModalProcessorCacheItem(*mm_item))\n'
                 '        return mm_item\n'),
                ('        from vllm import taaf_mm_processor_only\n'
                 '\n'
                 '        self.cache_if_fits(\n'
                 '            self._cache,\n'
                 '            mm_hash,\n'
                 '            MultiModalProcessorCacheItem(*taaf_mm_processor_only.compact(mm_item)),\n'
                 '        )\n'
                 '        return mm_item\n'),
            ),
        ),
    ),
}
