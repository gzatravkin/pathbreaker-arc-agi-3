TARGETS = {
    'vllm/taaf_gdn_accept.py': (
        '448f1aea63676b090bbd7603d7d916c6422c118b6d857da5ec32b8bfb90c7b46',
        (
            (
                ('import triton.language as tl\n'),
                ('import triton.language as tl\n'
                 '\n'
                 'from vllm import taaf_gdn_fused\n'),
            ),
            (
                ('    return records\n'
                 '\n'
                 '\n'
                 '@triton.jit(do_not_specialize=["num_rows"])\n'
                 'def _verify_kernel(\n'),
                ('    return taaf_gdn_fused.attach(layer, records)\n'
                 '\n'
                 '\n'
                 '@triton.jit(do_not_specialize=["num_rows"])\n'
                 'def _verify_kernel(\n'),
            ),
            (
                ('def verify_layer(layer, q, k, v, a, b, query_start_loc, state_indices, out=None):\n'),
                ('def verify_layer(layer, q, k, v, a, b, query_start_loc, state_indices, out=None):\n'
                 '    if taaf_gdn_fused.mode():\n'
                 '        return taaf_gdn_fused.verify_layer(layer, q, k, v, a, b, query_start_loc,\n'
                 '                                           state_indices, out)\n'),
            ),
            (
                ('def decode(layer, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata):\n'),
                ('def decode(layer, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata):\n'
                 '    if taaf_gdn_fused.mode():\n'
                 '        return taaf_gdn_fused.decode(layer, mixed_qkv, b, a, output_gate, '
                 'core_attn_out,\n'
                 '                                     attn_metadata)\n'),
            ),
            (
                ('def commit_for(builder, common, state_indices, query_start_loc, rows, '
                 'request_indices):\n'
                 '    context = builder.gdn_accept_context\n'),
                ('def commit_for(builder, common, state_indices, query_start_loc, rows, '
                 'request_indices):\n'
                 '    if taaf_gdn_fused.mode():\n'
                 '        return taaf_gdn_fused.commit_for(stock_commit_for, builder, common, '
                 'state_indices,\n'
                 '                                         query_start_loc, rows, request_indices)\n'
                 '    return stock_commit_for(builder, common, state_indices, query_start_loc, rows,\n'
                 '                            request_indices)\n'
                 '\n'
                 '\n'
                 'def stock_commit_for(builder, common, state_indices, query_start_loc, rows,\n'
                 '                     request_indices):\n'
                 '    context = builder.gdn_accept_context\n'),
            ),
            (
                ('def unit_reset(state, slot):\n'),
                ('def unit_reset(state, slot):\n'
                 '    taaf_gdn_fused.slot_reset(slot)\n'),
            ),
            (
                ('    state.taaf_left = (req_id, owners.pop(req_id, None))\n'),
                ('    state.taaf_left = (req_id, owners.pop(req_id, None))\n'
                 '    taaf_gdn_fused.slot_leave(state.taaf_left[1])\n'),
            ),
            (
                ('def slot_resume(slot, blocks, resumed, computed, state_idx, column):\n'
                 '    return None\n'),
                ('def slot_resume(slot, blocks, resumed, computed, state_idx, column):\n'
                 '    return taaf_gdn_fused.slot_resume(slot, blocks, resumed, computed, state_idx, column)\n'),
            ),
            (
                ('def unit_prepare(state, input_batch, attn_groups, num_reqs, for_capture):\n'),
                ('def unit_prepare(state, input_batch, attn_groups, num_reqs, for_capture):\n'
                 '    taaf_gdn_fused.slots_prepare(input_batch, attn_groups, for_capture)\n'),
            ),
        ),
    ),
}
