TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        None,
        (
            (
                ('from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd\n'),
                ('from vllm import taaf_gdn_fused as _taaf_gdn_q8\n'
                 '\n'
                 'fused_recurrent_gated_delta_rule_packed_decode = _taaf_gdn_q8.q8_inplace(\n'
                 '    fused_recurrent_gated_delta_rule_packed_decode)\n'
                 'fused_sigmoid_gating_delta_rule_update = _taaf_gdn_q8.q8_inplace(fused_sigmoid_gating_delta_rule_update)\n'
                 'from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd\n'),
            ),
            (
                ('            initial_state = ssm_state[prefill_state_indices]\n'),
                ('            initial_state = _taaf_gdn_q8.q8_gather(ssm_state, prefill_state_indices)\n'),
            ),
            (
                ('            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)\n'),
                ('            _taaf_gdn_q8.q8_scatter(ssm_state, prefill_state_indices, last_recurrent_state)\n'),
            ),
        ),
    ),
}
