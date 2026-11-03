TARGETS = {
    'vllm/model_executor/layers/fused_moe/experts/marlin_moe.py': (
        None,
        (
            ('    if workspace is None:\n        workspace = marlin_make_workspace_new(hidden_states.device, 4)\n',
             '    if workspace is None:\n        from vllm import taaf_small_kernels\n\n        workspace = taaf_small_kernels.marlin_workspace(hidden_states.device, 4, marlin_make_workspace_new)\n'),
        ),
    ),
}
