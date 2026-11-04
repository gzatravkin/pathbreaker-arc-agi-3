TARGETS = {
    'vllm/v1/worker/gpu/spec_decode/adaptive_verification.py': (
        '276b2fec56758a94c71b717f6786b8a6f126db2e20cf964350202feee61a0a39',
        (
            (
                '_PROFILE_REPLAYS = 5\n',
                '_PROFILE_REPLAYS = 5\n'
                '\n'
                '\n'
                'def _taaf_budget():\n'
                '    from vllm import taaf_draft_budget\n'
                '\n'
                '    return taaf_draft_budget if taaf_draft_budget.armed() else None\n',
            ),
            (
                '    def add_request(self, req_idx: int) -> None:\n',
                '    def add_request(self, req_idx: int) -> None:\n'
                '        _budget = _taaf_budget()\n'
                '        if _budget is not None:\n'
                '            _budget.reset(self, req_idx)\n'
                '            return\n',
            ),
            (
                '        timings to set_initial_cost_curves."""\n',
                '        timings to set_initial_cost_curves."""\n'
                '        _budget = _taaf_budget()\n'
                '        if _budget is not None:\n'
                '            _budget.note_capture_sizes(capture_sizes, self)\n'
                '            return\n',
            ),
            (
                '        def median_curve(\n'
                '            points: Iterable[tuple[int, float]],\n'
                '        ) -> list[tuple[int, float]]:\n',
                '        if _taaf_budget() is not None:\n'
                '            return\n'
                '\n'
                '        def median_curve(\n'
                '            points: Iterable[tuple[int, float]],\n'
                '        ) -> list[tuple[int, float]]:\n',
            ),
            (
                '        num_reqs = input_batch.num_reqs\n'
                '        ready_idx = self._stale_idx ^ 1\n',
                '        if _taaf_budget() is not None:\n'
                '            return\n'
                '        num_reqs = input_batch.num_reqs\n'
                '        ready_idx = self._stale_idx ^ 1\n',
            ),
            (
                '        assert self.cost_tables is not None\n'
                '        req_ids = list(num_tokens_per_req)\n',
                '        _budget = _taaf_budget()\n'
                '        if _budget is not None:\n'
                '            return _budget.get_num_tokens(self, num_tokens_per_req, draft_tokens)\n'
                '        assert self.cost_tables is not None\n'
                '        req_ids = list(num_tokens_per_req)\n',
            ),
            (
                '            if draft_budget < int(scheduled_drafts.sum()):\n'
                '                _assign_draft_token_budget_compiled(\n'
                '                    self._confidence_probs,\n'
                '                    idx_mapping,\n'
                '                    capacities,\n'
                '                    draft_budget,\n'
                '                    self.num_speculative_steps,\n'
                '                )\n',
                '            if draft_budget < int(scheduled_drafts.sum()):\n'
                '                _budget = _taaf_budget()\n'
                '                if _budget is not None:\n'
                '                    _budget.assign(self, idx_mapping, capacities, draft_budget, num_reqs)\n'
                '                else:\n'
                '                    _assign_draft_token_budget_compiled(\n'
                '                        self._confidence_probs,\n'
                '                        idx_mapping,\n'
                '                        capacities,\n'
                '                        draft_budget,\n'
                '                        self.num_speculative_steps,\n'
                '                    )\n',
            ),
        ),
    ),
    'vllm/v1/worker/gpu/spec_decode/rejection_sampler.py': (
        None,
        (
            (
                '        self.enable_adaptive_verification = spec_config.enable_adaptive_verification\n',
                '        self.enable_adaptive_verification = spec_config.enable_adaptive_verification and (\n'
                '            __import__("os").environ.get("TAAF_VLLM_DRAFT_BUDGET", "") in ("", "0")\n'
                '        )\n',
            ),
        ),
    ),
}
