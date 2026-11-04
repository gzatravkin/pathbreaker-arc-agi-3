TARGETS = {
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                ('            req_index = self.req_states.req_id_to_index[req_id]\n'
                 '            if self.adaptive_verification is not None:\n'
                 '                self.adaptive_verification.add_request(req_index)\n'),
                ('            req_index = self.req_states.req_id_to_index[req_id]\n'
                 '            from vllm import taaf_no_blank\n'
                 '\n'
                 '            taaf_no_blank.added(self, req_index)\n'
                 '            if self.adaptive_verification is not None:\n'
                 '                self.adaptive_verification.add_request(req_index)\n'),
            ),
            (
                ('        sampler_output: SamplerOutput | None\n'
                 '        if input_batch.num_reqs == 0:\n'),
                ('        from vllm import taaf_no_blank\n'
                 '\n'
                 '        taaf_no_blank_flags = taaf_no_blank.eager(self, logits, input_batch)\n'
                 '        sampler_output: SamplerOutput | None\n'
                 '        if input_batch.num_reqs == 0:\n'),
            ),
            (
                ('        if shard_metadata is not None:\n'
                 '            # Gather the sharded sampler outputs from the TP ranks into a single\n'),
                ('        taaf_no_blank.after(self, taaf_no_blank_flags, sampler_output, input_batch)\n'
                 '        if shard_metadata is not None:\n'
                 '            # Gather the sharded sampler outputs from the TP ranks into a single\n'),
            ),
        ),
    ),
}
