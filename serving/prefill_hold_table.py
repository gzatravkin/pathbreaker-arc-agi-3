TARGETS = {
    'vllm/v1/core/sched/scheduler.py': (
        None,
        (
            (
                ('            step_skipped_waiting = create_request_queue(self.policy)\n'
                 '\n'
                 '            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n'),
                ('            step_skipped_waiting = create_request_queue(self.policy)\n'
                 '\n'
                 '            from vllm import taaf_prefill_hold as _taaf_prefill_hold\n'
                 '\n'
                 '            _taaf_held = _taaf_prefill_hold.hold(self, scheduled_running_reqs)\n'
                 '            while (self.waiting or self.skipped_waiting) and token_budget > 0 and not _taaf_held:\n'),
            ),
        ),
    ),
}
