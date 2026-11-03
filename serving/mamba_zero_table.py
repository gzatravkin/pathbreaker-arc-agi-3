TARGETS = {
    'vllm/v1/core/single_type_kv_cache_manager.py': (
        None,
        (
            (
                ('            and isinstance(kv_cache_spec, AttentionSpec)\n'
                 '            and not isinstance(kv_cache_spec, CircularBufferSpec)\n'
                 '        )\n'),
                ('            and isinstance(kv_cache_spec, (AttentionSpec, MambaSpec))\n'
                 '            and not isinstance(kv_cache_spec, CircularBufferSpec)\n'
                 '        )\n'),
            ),
            (
                ('                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)\n'
                 '                returned_blocks = req_blocks[prev_block_len:]\n'),
                ('                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)\n'
                 '                if self._record_new_block_ids:\n'
                 '                    self.new_block_ids.extend(b.block_id for b in new_blocks)\n'
                 '                returned_blocks = req_blocks[prev_block_len:]\n'),
            ),
        ),
    ),
    'vllm/v1/worker/utils.py': (
        None,
        (
            (
                ('            spec = group.kv_cache_spec\n'
                 '            if not isinstance(spec, AttentionSpec):\n'
                 '                continue\n'),
                ('            spec = group.kv_cache_spec\n'
                 '            if isinstance(spec, MambaSpec):\n'
                 '                for layer_name in group.layer_names:\n'
                 '                    held = static_forward_context[layer_name].kv_cache\n'
                 '                    for state in held if isinstance(held, (list, tuple)) else (held,):\n'
                 '                        if not isinstance(state, torch.Tensor) or state.dim() < 2:\n'
                 '                            continue\n'
                 '                        if state.device.type != self.device.type or state.shape[0] != num_blocks:\n'
                 '                            mamba_zero_skipped += 1\n'
                 '                            continue\n'
                 '                        el = state.element_size()\n'
                 '                        stride = state.stride(0) * el\n'
                 '                        page = el + sum((state.shape[d] - 1) * state.stride(d) * el\n'
                 '                                        for d in range(1, state.dim()))\n'
                 '                        addr = state.data_ptr()\n'
                 '                        if stride % 4 or page % 4 or addr % 4 or page > stride:\n'
                 '                            mamba_zero_skipped += 1\n'
                 '                            continue\n'
                 '                        idx = seen_ptrs.get(addr)\n'
                 '                        if idx is not None and seg_block_strides[idx] == stride // 4:\n'
                 '                            seg_page_sizes[idx] = max(seg_page_sizes[idx], page // 4)\n'
                 '                            mamba_zero_merged += 1\n'
                 '                            continue\n'
                 '                        if idx is None:\n'
                 '                            seen_ptrs[addr] = len(seg_addrs)\n'
                 '                        seg_addrs.append(addr)\n'
                 '                        seg_block_strides.append(stride // 4)\n'
                 '                        seg_page_sizes.append(page // 4)\n'
                 '                        mamba_zero_added += 1\n'
                 '                continue\n'
                 '            if not isinstance(spec, AttentionSpec):\n'
                 '                continue\n'),
            ),
            (
                ('        for group in attn_groups_iter:\n'),
                ('        mamba_zero_added = mamba_zero_merged = mamba_zero_skipped = 0\n'
                 '        for group in attn_groups_iter:\n'),
            ),
            (
                ('        if not seg_addrs:\n'),
                ('        print(f"MAMBA_ZERO_NEW worker: {mamba_zero_added} Mamba state segment(s) join the new-block '
                 'zeroing, {mamba_zero_merged} widen an overlaid attention segment, {mamba_zero_skipped} skipped '
                 '(other device, block count or alignment); a block the scheduler hands a Mamba group starts as '
                 'zeros instead of the bytes its previous owner left", flush=True)\n'
                 '        if not seg_addrs:\n'),
            ),
        ),
    ),
}
