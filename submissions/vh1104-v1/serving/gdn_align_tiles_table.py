TARGETS = {
    'vllm/v1/worker/mamba_utils.py': (
        '23fdaa4e8f63ee8f938eaa1a2b39f803cb04fa6ecd69789563254b1a0691413a',
        (
            (
                ('        grid = (num_reqs, total_states, _TEMPORAL_TILES)\n'
                 '        precopy_mamba_align_fused_kernel[grid](\n'),
                ('        from vllm import taaf_gdn_align_tiles\n'
                 '\n'
                 '        _TEMPORAL_TILES = taaf_gdn_align_tiles.tiles(globals()["_TEMPORAL_TILES"])\n'
                 '        grid = (num_reqs, total_states, _TEMPORAL_TILES)\n'
                 '        precopy_mamba_align_fused_kernel[grid](\n'),
            ),
            (
                ('        grid = (num_reqs, total_states, _TEMPORAL_TILES)\n'
                 '        postprocess_mamba_fused_kernel[grid](\n'
                 '            num_accepted_tokens_snapshot,\n'),
                ('        from vllm import taaf_gdn_align_tiles\n'
                 '\n'
                 '        _TEMPORAL_TILES = taaf_gdn_align_tiles.tiles(globals()["_TEMPORAL_TILES"])\n'
                 '        grid = (num_reqs, total_states, _TEMPORAL_TILES)\n'
                 '        postprocess_mamba_fused_kernel[grid](\n'
                 '            num_accepted_tokens_snapshot,\n'),
            ),
        ),
    ),
}
