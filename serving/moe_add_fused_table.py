TARGETS = {
    'vllm/model_executor/layers/fused_moe/runner/moe_runner.py': (
        '05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c',
        (
            (
                ('        result = self._forward_entry(\n'
                 '            hidden_states,\n'
                 '            router_logits,\n'
                 '            shared_experts_input,\n'),
                ('        from vllm import taaf_moe_add_fused\n'
                 '\n'
                 '        taaf_moe_add_fused.arm(self, hidden_states)\n'
                 '        result = self._forward_entry(\n'
                 '            hidden_states,\n'
                 '            router_logits,\n'
                 '            shared_experts_input,\n'),
            ),
            (
                ('        if shared_output is not None:\n'
                 '            result = shared_output + fused_output\n'),
                ('        if shared_output is not None:\n'
                 '            result = taaf_moe_add_fused.add(self, shared_output, fused_output)\n'),
            ),
        ),
    ),
    'vllm/model_executor/models/qwen2_moe.py': (
        'cecb5fb8c82bc92743016a56bc3991f3bd612b9767063ccdc0715a49a3e54442',
        (
            (
                ('            out = F.sigmoid(self.expert_gate(x)[0]) * out\n'),
                ('            from vllm import taaf_moe_add_fused\n'
                 '\n'
                 '            out = taaf_moe_add_fused.shared_gate(self, x, out)\n'),
            ),
        ),
    ),
}
