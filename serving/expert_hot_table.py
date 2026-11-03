TARGETS = {
    'vllm/v1/worker/gpu_worker.py': (
        None,
        (
            (
                ('        # configure float32 matmul precision according to vLLM env.\n'
                 '        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION\n'),
                ('        from vllm import taaf_expert_hot as _taaf_expert_hot\n'
                 '        _taaf_expert_hot.install_worker()\n'
                 '        # configure float32 matmul precision according to vLLM env.\n'
                 '        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION\n'),
            ),
        ),
    ),
    'vllm/v1/engine/core.py': (
        None,
        (
            (
                ('        # plugins need to be loaded at the engine/scheduler level too\n'
                 '        from vllm.plugins import load_general_plugins\n'),
                ('        from vllm import taaf_expert_hot as _taaf_expert_hot\n'
                 '        _taaf_expert_hot.install_engine()\n'
                 '        # plugins need to be loaded at the engine/scheduler level too\n'
                 '        from vllm.plugins import load_general_plugins\n'),
            ),
        ),
    ),
}
