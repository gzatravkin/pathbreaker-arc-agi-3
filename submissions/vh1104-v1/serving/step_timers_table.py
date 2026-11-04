TARGETS = {
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                ('        get_offloader().post_init()\n'
                 '\n'
                 '    def get_model(self) -> nn.Module:\n'),
                ('        get_offloader().post_init()\n'
                 '        from vllm import taaf_step_timers\n'
                 '\n'
                 '        taaf_step_timers.install(self)\n'
                 '\n'
                 '    def get_model(self) -> nn.Module:\n'),
            ),
        ),
    ),
}
