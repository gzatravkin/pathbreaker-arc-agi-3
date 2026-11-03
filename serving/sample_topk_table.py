TARGETS = {
    'vllm/v1/worker/gpu/spec_decode/rejection_sampler.py': (
        None,
        (
            (
                ('    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n'
                 '        processed_logits = self.sampler.apply_sampling_params(\n'),
                ('    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n'
                 '        from vllm import taaf_sample_topk\n'
                 '\n'
                 '        taaf_compact = taaf_sample_topk.verify(\n'
                 '            self, logits, draft_logits, draft_sampled, pos, cu_num_logits, idx_mapping,\n'
                 '            idx_mapping_np, expanded_idx_mapping, expanded_local_pos)\n'
                 '        if taaf_compact is not None:\n'
                 '            return taaf_compact\n'
                 '        processed_logits = self.sampler.apply_sampling_params(\n'),
            ),
            (
                ('            # draft_logits uses persistent request-state indices and stays global.\n'
                 '            processed_logits, sampled, num_sampled = self._verify(\n'),
                ('            # draft_logits uses persistent request-state indices and stays global.\n'
                 '            self._taaf_cu_np = chunk_cu_num_logits_np\n'
                 '            processed_logits, sampled, num_sampled = self._verify(\n'),
            ),
        ),
    ),
}
