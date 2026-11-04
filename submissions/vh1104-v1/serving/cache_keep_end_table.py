TARGETS = {
    'vllm/v1/core/kv_cache_manager.py': (
        '58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa',
        (
            (
                'from vllm.v1.request import Request, RequestStatus\n',
                ('from vllm.v1.request import Request, RequestStatus\n'
                 'from vllm import taaf_cache_keep_end\n'),
            ),
            (
                ('        self.empty_kv_cache_blocks = KVCacheBlocks(\n'
                 '            tuple(() for _ in range(self.num_kv_cache_groups))\n'
                 '        )\n'
                 '\n'
                 '    @property\n'
                 '    def usage(self) -> float:\n'),
                ('        self.empty_kv_cache_blocks = KVCacheBlocks(\n'
                 '            tuple(() for _ in range(self.num_kv_cache_groups))\n'
                 '        )\n'
                 '        self.taaf_unit = taaf_cache_keep_end.ReplyTails.attach(self)\n'
                 '\n'
                 '    @property\n'
                 '    def usage(self) -> float:\n'),
            ),
            (
                '    def free(self, request: Request) -> None:\n',
                ('    def taaf_unit_targets(self, scheduler):\n'
                 '        if self.taaf_unit is None:\n'
                 '            return None\n'
                 '        return self.taaf_unit.on_schedule(scheduler)\n'
                 '\n'
                 '    def free(self, request: Request) -> None:\n'),
            ),
            (
                '        self.coordinator.free(request.request_id)\n',
                ('        released = [] if self.taaf_unit is None else self.taaf_unit.on_free(request)\n'
                 '        self.coordinator.free(request.request_id)\n'
                 '        if released:\n'
                 '            self.block_pool.free_blocks(released)\n'),
            ),
            (
                '        return self.coordinator.pop_blocks_for_free(request.request_id)\n',
                ('        released = [] if self.taaf_unit is None else self.taaf_unit.on_free(request)\n'
                 '        return [*released, *self.coordinator.pop_blocks_for_free(request.request_id)]\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                'from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput\n',
                ('from vllm import taaf_gdn_accept\n'
                 'from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput\n'),
            ),
            (
                ('                scheduler_output.kv_cache_block_copies,\n'
                 '            )\n'),
                ('                scheduler_output.kv_cache_block_copies,\n'
                 '            )\n'
                 '        taaf_gdn_accept.unit_receive(self, scheduler_output)\n'),
            ),
        ),
    ),
    'vllm/v1/core/sched/scheduler.py': (
        None,
        (
            (
                '        # NOTE(Kuntai): this function is designed for multiple purposes:\n',
                ('        scheduler_output.taaf_unit_targets = (\n'
                 '            self.kv_cache_manager.taaf_unit_targets(self)\n'
                 '        )\n'
                 '        # NOTE(Kuntai): this function is designed for multiple purposes:\n'),
            ),
        ),
    ),
}
