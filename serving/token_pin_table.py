TARGETS = {
    'vllm/v1/worker/gpu/states.py': (
        '99418f5df43ca612ded72609fee011620b065b2cab2f249ccb387096bf4ae71a',
        (
            (
                ('        self.next_prefill_tokens = torch.zeros(\n'
                 '            num_prefill_lookahead,\n'
                 '            self.max_num_reqs,\n'
                 '            dtype=torch.int32,\n'
                 '            device=device,\n'
                 '        )\n'),
                ('        self.next_prefill_tokens = torch.zeros(\n'
                 '            num_prefill_lookahead,\n'
                 '            self.max_num_reqs,\n'
                 '            dtype=torch.int32,\n'
                 '            device=device,\n'
                 '        )\n'
                 '        from vllm import taaf_token_pin\n'
                 '\n'
                 '        taaf_token_pin.setup(self.max_num_reqs, device)\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                ('                self.adaptive_verification.add_request(req_index)\n'),
                ('                self.adaptive_verification.add_request(req_index)\n'
                 '            from vllm import taaf_token_pin\n'
                 '\n'
                 '            taaf_token_pin.admit(self, req_index, new_req_data.prefill_token_ids)\n'),
            ),
            (
                ('            input_batch = self.prepare_inputs(\n'
                 '                scheduler_output, batch_req_state, batch_desc\n'
                 '            )\n'
                 '            block_tables, slot_mappings = self.prepare_attn(input_batch)\n'),
                ('            input_batch = self.prepare_inputs(\n'
                 '                scheduler_output, batch_req_state, batch_desc\n'
                 '            )\n'
                 '            from vllm import taaf_token_pin\n'
                 '\n'
                 '            taaf_token_pin.gather(input_batch.idx_mapping, input_batch.num_reqs)\n'
                 '            block_tables, slot_mappings = self.prepare_attn(input_batch)\n'),
            ),
        ),
    ),
    'vllm/models/qwen4_exp/nvidia/indexer_qsa.py': (
        '23a39cc4ee463bd3441e4343d9a0933ab1a81ac5a3f5a5af80f5b084f70d8fef',
        (
            (
                ('        num_decode_tokens = compressed_metadata.num_decode_tokens\n'
                 '        decode_query_len = compressed_metadata.decode_query_len\n'),
                ('        num_decode_tokens = compressed_metadata.num_decode_tokens\n'
                 '        decode_query_len = compressed_metadata.decode_query_len\n'
                 '        from vllm import taaf_token_pin\n'
                 '\n'
                 '        taaf_token_pin.context(raw_metadata.token_to_req, num_decode_tokens)\n'),
            ),
        ),
    ),
    'vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py': (
        '2c34883a6396bac0238f6308196854b653eadab8bbebbe7dd2d558e30d065cbb',
        (
            (
                ('    _topk(\n'
                 '        logits,\n'
                 '        visible_blocks,\n'
                 '        token_topk,\n'),
                ('    from vllm import taaf_token_pin\n'
                 '\n'
                 '    taaf_token_pin.bump(logits, visible_blocks)\n'
                 '    _topk(\n'
                 '        logits,\n'
                 '        visible_blocks,\n'
                 '        token_topk,\n'),
            ),
            (
                ('        _topk(\n'
                 '            logits,\n'
                 '            visible_blocks[query_slice],\n'),
                ('        from vllm import taaf_token_pin\n'
                 '\n'
                 '        taaf_token_pin.bump(logits, visible_blocks[query_slice], query_start)\n'
                 '        _topk(\n'
                 '            logits,\n'
                 '            visible_blocks[query_slice],\n'),
            ),
        ),
    ),
}
