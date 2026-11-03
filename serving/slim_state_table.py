RIDE = ('vllm/v1/kv_cache_interface.py',)

TARGETS = {
    'vllm/v1/kv_cache_interface.py': (
        '1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb',
        (
            (
                ('            and spec.page_size_bytes == self.page_size_bytes\n'
                 '            and spec.tp_replicated == self.tp_replicated\n'
                 '            for spec in kv_cache_specs.values()\n'),
                ('            and spec.page_size_bytes == self.page_size_bytes\n'
                 '            for spec in kv_cache_specs.values()\n'),
            ),
        ),
    ),
    'vllm/v1/core/sched/scheduler.py': (
        None,
        (
            (
                'from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager\n',
                ('from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager\n'
                 'from vllm import taaf_slim_state\n'),
            ),
            (
                '        self.kv_cache_manager.new_step_starts()\n',
                ('        self.kv_cache_manager.new_step_starts()\n'
                 '        taaf_slim_state.on_step(self)\n'),
            ),
            (
                ('                    # The request cannot be scheduled.\n'
                 '                    # Preempt the lowest-priority request.\n'),
                ('                    if taaf_slim_state.make_room(self):\n'
                 '                        continue\n'
                 '\n'
                 '                    # The request cannot be scheduled.\n'
                 '                    # Preempt the lowest-priority request.\n'),
            ),
        ),
    ),
}
