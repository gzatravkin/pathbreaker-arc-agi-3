TARGETS = {
    'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py': (
        '214b61fa0abe88573beb779872c035b3954e5651a85bd438d99ee5905d3a62ec',
        (
            (
                'from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand32\n',
                ('from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand32\n'
                 'from vllm.taaf_accept_lambda import rows as _taaf_lambda_rows\n'
                 'from vllm.taaf_accept_lambda import settle as _taaf_lambda_settle\n'),
            ),
            (
                ('    temp_ptr,\n'
                 '    vocab_num_blocks,\n'
                 '    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n'
                 '    HAS_DRAFT_LOGITS: tl.constexpr,\n'
                 '):\n'
                 '    req_idx = tl.program_id(0)\n'),
                ('    temp_ptr,\n'
                 '    taaf_lambda_ptr,\n'
                 '    vocab_num_blocks,\n'
                 '    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n'
                 '    HAS_DRAFT_LOGITS: tl.constexpr,\n'
                 '):\n'
                 '    req_idx = tl.program_id(0)\n'),
            ),
            (
                '            log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)\n',
                ('            log_p = tl.minimum(\n'
                 '                log_p\n'
                 '                + (target_logprob - draft_logprob)\n'
                 '                + tl.load(taaf_lambda_ptr + logit_idx),\n'
                 '                0.0,\n'
                 '            )\n'),
            ),
            (
                ('    # [num_speculative_steps]\n'
                 '    synthetic_conditional_rates_ptr,\n'),
                ('    # [num_speculative_steps]\n'
                 '    synthetic_conditional_rates_ptr,\n'
                 '    taaf_lambda_ptr,\n'),
            ),
            (
                '                    accepted = target_logprob > tl.log(u) + draft_logprob\n',
                ('                    accepted = (\n'
                 '                        target_logprob + tl.load(taaf_lambda_ptr + logit_idx)\n'
                 '                        > tl.log(u) + draft_logprob\n'
                 '                    )\n'),
            ),
            (
                ('    # Precompute the running joint ratio and residual mass for block\n'
                 '    # verification.\n'),
                ('    taaf_lambda = _taaf_lambda_rows(\n'
                 '        draft_sampled,\n'
                 '        cu_num_logits,\n'
                 '        idx_mapping,\n'
                 '        temperature,\n'
                 '        num_logits,\n'
                 '        num_speculative_steps,\n'
                 '    )\n'
                 '\n'
                 '    # Precompute the running joint ratio and residual mass for block\n'
                 '    # verification.\n'),
            ),
            (
                ('            idx_mapping,\n'
                 '            temperature,\n'
                 '            vocab_num_blocks,\n'),
                ('            idx_mapping,\n'
                 '            temperature,\n'
                 '            taaf_lambda,\n'
                 '            vocab_num_blocks,\n'),
            ),
            (
                ('        synthetic_conditional_rates,\n'
                 '        cumulative_log_p,\n'),
                ('        synthetic_conditional_rates,\n'
                 '        taaf_lambda,\n'
                 '        cumulative_log_p,\n'),
            ),
            (
                '    # Insert the resampled tokens into the output sampled.\n',
                ('    _taaf_lambda_settle(num_sampled, cu_num_logits, idx_mapping, temperature)\n'
                 '\n'
                 '    # Insert the resampled tokens into the output sampled.\n'),
            ),
        ),
    ),
}
