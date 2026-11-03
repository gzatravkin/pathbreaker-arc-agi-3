TARGETS = {
    'vllm/v1/core/sched/scheduler.py': (
        None,
        (
            (
                ('        return (\n'
                 '            encoder_inputs_to_schedule,\n'
                 '            num_new_tokens,\n'
                 '            encoder_compute_budget,\n'
                 '            external_load_encoder_input,\n'
                 '        )\n'),
                ('        from vllm import taaf_vision_ahead\n'
                 '\n'
                 '        num_new_tokens = taaf_vision_ahead.hold(\n'
                 '            self, request, num_computed_tokens, num_new_tokens, shift_computed_tokens,\n'
                 '            encoder_inputs_to_schedule,\n'
                 '        )\n'
                 '        return (\n'
                 '            encoder_inputs_to_schedule,\n'
                 '            num_new_tokens,\n'
                 '            encoder_compute_budget,\n'
                 '            external_load_encoder_input,\n'
                 '        )\n'),
            ),
            (
                ('            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),\n'),
                ('            free_encoder_mm_hashes=__import__("vllm.taaf_vision_ahead", fromlist=["freed"]).freed(\n'
                 '                self, self.encoder_cache_manager.get_freed_mm_hashes()),\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/model_states/default.py': (
        'bb614f814780f052e869f9acf31dfbfa68718ee7dc6252eefa901b5730ee8a2a',
        (
            (
                ('            self.execute_mm_encoder(scheduled_encoder_inputs)\n'),
                ('            from vllm import taaf_vision_ahead\n'
                 '\n'
                 '            taaf_vision_ahead.execute(self, scheduled_encoder_inputs, input_batch)\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/mm/encoder_runner.py': (
        '0981e352d70301b7154a217a48665ffe35f215d053ec5740c2515a3e1e11b4e6',
        (
            (
                ('        return mm_embeds, is_mm_embed\n'),
                ('        from vllm import taaf_vision_ahead\n'
                 '\n'
                 '        taaf_vision_ahead.settle(mm_embeds)\n'
                 '        return mm_embeds, is_mm_embed\n'),
            ),
        ),
    ),
}
