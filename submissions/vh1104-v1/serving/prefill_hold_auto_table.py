TARGETS = {
    'vllm/v1/core/sched/scheduler.py': (
        None,
        (
            (
                ('        # First, schedule the RUNNING requests.\n'
                 '        req_index = 0\n'),
                ('        from vllm import taaf_prefill_hold as _taaf_prefill_hold\n'
                 '\n'
                 '        _taaf_hold_step = _taaf_prefill_hold.plan(self, token_budget)\n'
                 '\n'
                 '        # First, schedule the RUNNING requests.\n'
                 '        req_index = 0\n'),
            ),
            (
                ('            if defer_prefills and request.is_prefill_chunk:\n'
                 '                # DP prefill balancing: defer this in-progress prefill chunk to a\n'
                 '                # cadence-aligned step; decodes still run to fill this step.\n'
                 '                req_index += 1\n'
                 '                continue\n'),
                ('            if defer_prefills and request.is_prefill_chunk:\n'
                 '                # DP prefill balancing: defer this in-progress prefill chunk to a\n'
                 '                # cadence-aligned step; decodes still run to fill this step.\n'
                 '                req_index += 1\n'
                 '                continue\n'
                 '\n'
                 '            if _taaf_hold_step and request.is_prefill_chunk:\n'
                 '                req_index += 1\n'
                 '                continue\n'),
            ),
            (
                ('        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n'),
                ('        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n'
                 '        _taaf_prefill_hold.scheduled(self, num_scheduled_tokens, scheduled_new_reqs,\n'
                 '                                     scheduled_resumed_reqs, total_num_scheduled_tokens)\n'),
            ),
        ),
    ),
}
