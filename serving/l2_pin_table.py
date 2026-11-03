TARGETS = {
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                ('        compilation_counter.num_gpu_runner_capture_triggers += 1\n'),
                ('        compilation_counter.num_gpu_runner_capture_triggers += 1\n'
                 '        from vllm import taaf_l2_pin\n'
                 '\n'
                 '        taaf_l2_pin.apply(self.model, getattr(self, "speculator", None))\n'),
            ),
        ),
    ),
}
