TARGETS = {
    'vllm/v1/worker/gpu/cudagraph_utils.py': (
        '6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff',
        (
            (
                ('                descs_by_mode[mixed_mode].append(desc)\n'
                 '\n'
                 '                ubatch_desc = self._maybe_ubatch_twin(desc)\n'
                 '                if ubatch_desc is not None:\n'
                 '                    descs_by_mode[mixed_mode].append(ubatch_desc)\n'),
                ('                descs_by_mode[mixed_mode].append(desc)\n'
                 '\n'
                 '                ubatch_desc = self._maybe_ubatch_twin(desc)\n'
                 '                if ubatch_desc is not None:\n'
                 '                    descs_by_mode[mixed_mode].append(ubatch_desc)\n'
                 '\n'
                 '                if mixed_mode == CUDAGraphMode.PIECEWISE:\n'
                 '                    from vllm import taaf_mixed_full_graph\n'
                 '\n'
                 '                    descs_by_mode[CUDAGraphMode.FULL].extend(\n'
                 '                        taaf_mixed_full_graph.mixed_descriptors(\n'
                 '                            self,\n'
                 '                            num_tokens,\n'
                 '                            num_active_loras,\n'
                 '                            BatchExecutionDescriptor,\n'
                 '                            CUDAGraphMode.FULL,\n'
                 '                        )\n'
                 '                    )\n'),
            ),
            (
                ('            for desc in self._candidates[key]:\n'
                 '                if _is_compatible(\n'),
                ('            for desc in self._candidates[key]:\n'
                 '                from vllm import taaf_mixed_full_graph\n'
                 '\n'
                 '                if taaf_mixed_full_graph.refuse(self, desc, CUDAGraphMode.FULL):\n'
                 '                    continue\n'
                 '                if _is_compatible(\n'),
            ),
            (
                ('            attn_metadata, slot_mappings = prepare_inputs_to_capture(\n'
                 '                num_reqs,\n'),
                ('            from vllm import taaf_mixed_full_graph\n'
                 '\n'
                 '            attn_metadata, slot_mappings = prepare_inputs_to_capture(\n'
                 '                num_reqs,\n'),
            ),
            (
                ('                max_query_len=desc.max_query_len,\n'
                 '                pcp_manager=pcp_manager,\n'
                 '            )\n'),
                ('                max_query_len=desc.max_query_len,\n'
                 '                pcp_manager=pcp_manager,\n'
                 '                mixed_bucket=taaf_mixed_full_graph.is_mixed(desc, CUDAGraphMode.FULL),\n'
                 '                decode_query_len=self.decode_query_len,\n'
                 '            )\n'),
            ),
            (
                ('    pcp_manager: "PCPManager | None" = None,\n'
                 ') -> AttentionState:\n'
                 '    input_batch = InputBatch.make_dummy(\n'
                 '        num_reqs, num_tokens, input_buffers, max_query_len=max_query_len\n'
                 '    )\n'),
                ('    pcp_manager: "PCPManager | None" = None,\n'
                 '    mixed_bucket: bool = False,\n'
                 '    decode_query_len: int = 1,\n'
                 ') -> AttentionState:\n'
                 '    from vllm import taaf_mixed_full_graph\n'
                 '\n'
                 '    input_batch = taaf_mixed_full_graph.dummy_batch(\n'
                 '        InputBatch,\n'
                 '        num_reqs,\n'
                 '        num_tokens,\n'
                 '        input_buffers,\n'
                 '        max_query_len,\n'
                 '        mixed_bucket,\n'
                 '        decode_query_len,\n'
                 '    )\n'),
            ),
            (
                ('                descs = self._capture_descs[mode]\n'),
                ('                descs = self._capture_descs[mode]\n'
                 '                if mode == CUDAGraphMode.FULL:\n'
                 '                    from vllm import taaf_mixed_full_graph\n'
                 '\n'
                 '                    descs = taaf_mixed_full_graph.screen(\n'
                 '                        self, descs, create_forward_fn, CUDAGraphMode.FULL\n'
                 '                    )\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                ('        cudagraph_stats = None\n'
                 '        if not dummy_run:\n'
                 '            # Common case.\n'),
                ('        from vllm import taaf_mixed_full_graph\n'
                 '\n'
                 '        batch_desc = taaf_mixed_full_graph.settle(\n'
                 '            self, batch_req_state, batch_desc, scheduler_output\n'
                 '        )\n'
                 '        cudagraph_stats = None\n'
                 '        if not dummy_run:\n'
                 '            # Common case.\n'),
            ),
        ),
    ),
}
