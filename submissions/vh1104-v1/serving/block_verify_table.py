TARGETS = {
    'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py': (
        '214b61fa0abe88573beb779872c035b3954e5651a85bd438d99ee5905d3a62ec',
        (
            (
                'from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax\n',
                ('from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax\n'
                 '\n'
                 'from vllm.taaf_block_verify import U_SALT as _TAAF_U_SALT_VALUE\n'
                 'from vllm.taaf_block_verify import sparse_residual as _taaf_sparse_residual\n'
                 'from vllm.taaf_block_verify import tally as _taaf_block_tally\n'
                 'from vllm.taaf_block_verify import tick as _taaf_block_tick\n'
                 '\n'
                 '_TAAF_U_SALT = tl.constexpr(_TAAF_U_SALT_VALUE)\n'),
            ),
            (
                ('    local_residual_mass_stride,\n'
                 '    vocab_num_blocks,\n'
                 '    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n'
                 '    HAS_DRAFT_LOGITS: tl.constexpr,\n'
                 '    SYNTHETIC_MODE: tl.constexpr,\n'),
                ('    local_residual_mass_stride,\n'
                 '    taaf_tally_ptr,\n'
                 '    vocab_num_blocks,\n'
                 '    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n'
                 '    HAS_DRAFT_LOGITS: tl.constexpr,\n'
                 '    SYNTHETIC_MODE: tl.constexpr,\n'),
            ),
            (
                ('    accepted_length = tl.zeros((), tl.int64)\n'
                 '    target_lse = 0.0\n'
                 '    draft_lse = 0.0\n'
                 '    verifying = True\n'),
                ('    accepted_length = tl.zeros((), tl.int64)\n'
                 '    target_lse = 0.0\n'
                 '    draft_lse = 0.0\n'
                 '    verifying = True\n'
                 '    taaf_token_length = tl.zeros((), tl.int64)\n'
                 '    taaf_token_alive = tl.full((), 1, tl.int64)\n'),
            ),
            (
                ('            elif USE_BLOCK_VERIFICATION:\n'
                 '                # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444\n'),
                ('            elif USE_BLOCK_VERIFICATION:\n'
                 '                # Block verification (Sun et al., 2024): https://arxiv.org/abs/2403.10444\n'
                 '                u = tl_rand32(seed + (i + 1) * _TAAF_U_SALT, pos, includes_zero=False)\n'
                 '                taaf_target_logprob, taaf_draft_logprob, _, _ = (\n'
                 '                    _compute_global_logprobs_and_logsumexp(\n'
                 '                        draft_sampled,\n'
                 '                        True,\n'
                 '                        logit_idx,\n'
                 '                        req_state_idx,\n'
                 '                        i,\n'
                 '                        temp,\n'
                 '                        target_logits_ptr,\n'
                 '                        target_logits_stride,\n'
                 '                        target_local_max_ptr,\n'
                 '                        target_local_max_stride,\n'
                 '                        target_local_sumexp_ptr,\n'
                 '                        target_local_sumexp_stride,\n'
                 '                        draft_logits_ptr,\n'
                 '                        draft_logits_stride_0,\n'
                 '                        draft_logits_stride_1,\n'
                 '                        draft_local_max_ptr,\n'
                 '                        draft_local_max_stride,\n'
                 '                        draft_local_sumexp_ptr,\n'
                 '                        draft_local_sumexp_stride,\n'
                 '                        vocab_num_blocks,\n'
                 '                        PADDED_VOCAB_NUM_BLOCKS,\n'
                 '                        HAS_DRAFT_LOGITS,\n'
                 '                    )\n'
                 '                )\n'
                 '                taaf_token_alive = taaf_token_alive * (\n'
                 '                    taaf_target_logprob > tl.log(u) + taaf_draft_logprob\n'
                 '                ).to(tl.int64)\n'
                 '                taaf_token_length += taaf_token_alive\n'),
            ),
            (
                ('    tl.store(rejected_steps_ptr + req_idx, accepted_length)\n'
                 '    if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length < num_draft_tokens:\n'),
                ('    if USE_BLOCK_VERIFICATION:\n'
                 '        if not is_greedy and num_draft_tokens > 0:\n'
                 '            one = tl.full((), 1, tl.int64)\n'
                 '            tl.atomic_add(taaf_tally_ptr + 0, one)\n'
                 '            tl.atomic_add(taaf_tally_ptr + 1, accepted_length)\n'
                 '            tl.atomic_add(taaf_tally_ptr + 2, taaf_token_length)\n'
                 '            tl.atomic_add(taaf_tally_ptr + 3, (accepted_length > taaf_token_length).to(tl.int64))\n'
                 '            tl.atomic_add(taaf_tally_ptr + 4, (accepted_length < taaf_token_length).to(tl.int64))\n'
                 '            tl.atomic_add(taaf_tally_ptr + 5, (accepted_length == num_draft_tokens).to(tl.int64))\n'
                 '            tl.atomic_add(taaf_tally_ptr + 6, (taaf_token_length == num_draft_tokens).to(tl.int64))\n'
                 '    tl.store(rejected_steps_ptr + req_idx, accepted_length)\n'
                 '    if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length < num_draft_tokens:\n'),
            ),
            (
                ('            local_residual_mass = target_logits.new_empty(\n'
                 '                num_logits, vocab_num_blocks, dtype=torch.float32\n'
                 '            )\n'
                 '            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](\n'
                 '                local_residual_mass,\n'
                 '                local_residual_mass.stride(0),\n'
                 '                cumulative_log_p,\n'
                 '                target_logits,\n'
                 '                target_logits.stride(0),\n'
                 '                target_local_max,\n'
                 '                target_local_max.stride(0),\n'
                 '                target_local_sumexp,\n'
                 '                target_local_sumexp.stride(0),\n'
                 '                draft_logits,\n'
                 '                draft_logits_stride_0,\n'
                 '                draft_logits_stride_1,\n'
                 '                draft_local_max,\n'
                 '                draft_local_max.stride(0),\n'
                 '                draft_local_sumexp,\n'
                 '                draft_local_sumexp.stride(0),\n'
                 '                draft_sampled,\n'
                 '                expanded_idx_mapping,\n'
                 '                expanded_local_pos,\n'
                 '                temperature,\n'
                 '                vocab_size,\n'
                 '                num_speculative_steps,\n'
                 '                vocab_num_blocks,\n'
                 '                BLOCK_SIZE=VOCAB_BLOCK_SIZE,\n'
                 '                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,\n'
                 '            )\n'),
                ('            local_residual_mass = target_logits.new_zeros(\n'
                 '                num_logits, vocab_num_blocks, dtype=torch.float32\n'
                 '            )\n'
                 '            _taaf_sparse_residual(\n'
                 '                local_residual_mass,\n'
                 '                cumulative_log_p,\n'
                 '                target_logits,\n'
                 '                target_local_max,\n'
                 '                target_local_sumexp,\n'
                 '                draft_logits,\n'
                 '                draft_local_max,\n'
                 '                draft_local_sumexp,\n'
                 '                draft_sampled,\n'
                 '                expanded_idx_mapping,\n'
                 '                expanded_local_pos,\n'
                 '                temperature,\n'
                 '                vocab_size,\n'
                 '                num_speculative_steps,\n'
                 '                vocab_num_blocks,\n'
                 '                padded_vocab_num_blocks,\n'
                 '            )\n'),
            ),
            (
                ('        local_residual_mass.stride(0) if local_residual_mass is not None else 0,\n'
                 '        vocab_num_blocks,\n'),
                ('        local_residual_mass.stride(0) if local_residual_mass is not None else 0,\n'
                 '        _taaf_block_tally(target_logits.device),\n'
                 '        vocab_num_blocks,\n'),
            ),
            (
                ('        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,\n'
                 '    )\n'
                 '    return sampled, num_sampled\n'),
                ('        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,\n'
                 '    )\n'
                 '    _taaf_block_tick()\n'
                 '    return sampled, num_sampled\n'),
            ),
        ),
    ),
}
