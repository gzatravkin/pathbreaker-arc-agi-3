TARGETS = {
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                ('        if batch_desc.num_tokens == 0:\n'
                 '            # All DP ranks have zero tokens to run.\n'),
                ('        from vllm import taaf_mixed_full\n'
                 '\n'
                 '        taaf_mixed_full.observe(batch_req_state, batch_desc, self.decode_query_len)\n'
                 '        if batch_desc.num_tokens == 0:\n'
                 '            # All DP ranks have zero tokens to run.\n'),
            ),
            (
                ('        shard_metadata = None\n'
                 '        global_input_batch = input_batch\n'),
                ('        from vllm import taaf_mixed_full_spec\n'
                 '\n'
                 '        taken = taaf_mixed_full_spec.take(self, hidden_states, input_batch, grammar_output)\n'
                 '        if taken is not None:\n'
                 '            return taken\n'
                 '        shard_metadata = None\n'
                 '        global_input_batch = input_batch\n'),
            ),
        ),
    ),
}
