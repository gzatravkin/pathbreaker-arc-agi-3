TARGETS = {
    'vllm/model_executor/layers/fused_moe/experts/marlin_moe.py': (
        '93916b721548e6194dc70cf6cec224b4b1456a0cc0980235e97beba7bf36541d',
        (
            (
                ('        block_size_m = max(block_size_m, 16)\n'
                 '\n'
                 '    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(\n'),
                ('        block_size_m = max(block_size_m, 16)\n'
                 '\n'
                 '    from vllm import taaf_moe_block as _taaf_moe_block\n'
                 '    block_size_m = _taaf_moe_block.choose(hidden_states.size(0), block_size_m, locals(), fused_marlin_moe)\n'
                 '\n'
                 '    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(\n'),
            ),
            (
                ('        gate_up_input, a_scales1 = marlin_quant_input(hidden_states, input_dtype)\n'
                 '        if input_global_scale1 is not None:\n'
                 '            a_scales1 = a_scales1 * input_global_scale1\n'),
                ('        from vllm import taaf_moe_block as _taaf_moe_block\n'
                 '        gate_up_input, a_scales1 = _taaf_moe_block.quant_int8(hidden_states, input_global_scale1)\n'),
            ),
            (
                ('        intermediate_cache2, a_scales2 = marlin_quant_input(\n'
                 '            intermediate_cache2, input_dtype\n'
                 '        )\n'
                 '        if input_global_scale2 is not None:\n'
                 '            a_scales2 = a_scales2 * input_global_scale2\n'),
                ('        from vllm import taaf_moe_block as _taaf_moe_block\n'
                 '        intermediate_cache2, a_scales2 = _taaf_moe_block.quant_int8(intermediate_cache2,\n'
                 '                                                                    input_global_scale2, locals())\n'),
            ),
            (
                ('    intermediate_cache1 = ops.moe_wna16_marlin_gemm(\n'
                 '        gate_up_input,\n'),
                ('    from vllm import taaf_moe_block as _taaf_moe_block\n'
                 '    _taaf_act = _taaf_moe_block.silu_w1(locals())\n'
                 '    intermediate_cache1 = _taaf_moe_block.silu_gemm(_taaf_act, ops.moe_wna16_marlin_gemm)(\n'
                 '        gate_up_input,\n'),
            ),
            (
                ('    activation_input = intermediate_cache1.view(-1, w13_num_shards * N)\n'
                 '    if activation_func is None:\n'),
                ('    activation_input = None if _taaf_act is not None else intermediate_cache1.view(\n'
                 '        -1, w13_num_shards * N)\n'
                 '    if _taaf_act is not None:\n'
                 '        pass\n'
                 '    elif _taaf_moe_block.act_quant(locals()):\n'
                 '        pass\n'
                 '    elif activation_func is None:\n'),
            ),
            (
                ('    output = ops.moe_wna16_marlin_gemm(\n'
                 '        intermediate_cache2,\n'),
                ('    from vllm import taaf_moe_block as _taaf_moe_block\n'
                 '    _taaf_summed = _taaf_moe_block.sum_epilogue(locals())\n'
                 '    if _taaf_summed is not None:\n'
                 '        return _taaf_summed\n'
                 '\n'
                 '    output = ops.moe_wna16_marlin_gemm(\n'
                 '        intermediate_cache2,\n'),
            ),
            (
                ('    if output is None:\n'
                 '        output = torch.empty_like(hidden_states)\n'
                 '\n'
                 '    if moe_sum is None:\n'),
                ('    if output is None:\n'
                 '        output = torch.empty_like(hidden_states)\n'
                 '\n'
                 '    from vllm import taaf_moe_block as _taaf_moe_block\n'
                 '    if _taaf_moe_block.take_summed(output):\n'
                 '        return output\n'
                 '\n'
                 '    if moe_sum is None:\n'),
            ),
        ),
    ),
}
