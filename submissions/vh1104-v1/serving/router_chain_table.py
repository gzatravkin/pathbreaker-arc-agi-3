TARGETS = {
    'vllm/model_executor/layers/fused_moe/runner/moe_runner.py': (
        '05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c',
        (
            (
                ('        shared_experts_overlapping = False\n'
                 '        if self._shared_experts is not None:\n'
                 '            shared_experts_overlapping = self._shared_experts.maybe_forward_async(\n'),
                ('        from vllm import taaf_router_chain as _taaf_router_chain\n'
                 '\n'
                 '        _taaf_router_chain.begin(self, hidden_states, shared_experts_input)\n'
                 '        shared_experts_overlapping = False\n'
                 '        if self._shared_experts is not None:\n'
                 '            shared_experts_overlapping = self._shared_experts.maybe_forward_async(\n'),
            ),
            (
                ('            return self._maybe_combine(\n'
                 '                shared_output,\n'
                 '                hidden_states,\n'
                 '            )\n'),
                ('            _taaf_router_chain.end(self)\n'
                 '            return self._maybe_combine(\n'
                 '                shared_output,\n'
                 '                hidden_states,\n'
                 '            )\n'),
            ),
        ),
    ),
}
