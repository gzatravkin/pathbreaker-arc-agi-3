TARGETS = {
    'vllm/model_executor/layers/fused_moe/experts/marlin_moe.py': (
        '93916b721548e6194dc70cf6cec224b4b1456a0cc0980235e97beba7bf36541d',
        (
            (
                ('from vllm.scalar_type import ScalarType, scalar_types\n'),
                ('from vllm.scalar_type import ScalarType, scalar_types\n'
                 'from vllm import taaf_moe_taaf as _taaf_moe_taaf\n'),
            ),
            (
                ('        ctx = self._lora_context\n'
                 '        if ctx is None:\n'
                 '            fused_marlin_moe(\n'),
                ('        ctx = self._lora_context\n'
                 '        if ctx is None and _taaf_moe_taaf.routed(\n'
                 '            self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,\n'
                 '            global_num_experts, expert_map, workspace13, workspace2,\n'
                 '            apply_router_weight_on_input,\n'
                 '        ):\n'
                 '            return\n'
                 '        if ctx is None:\n'
                 '            fused_marlin_moe(\n'),
            ),
        ),
    ),
}
