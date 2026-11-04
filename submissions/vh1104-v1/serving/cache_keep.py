TARGETS = {
    'vllm/v1/core/kv_cache_utils.py': (
        '2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2',
        (
            (
                ('        groups,\n'
                 '        use_deepseek_v4_fallback=_is_deepseek_v4_eagle(vllm_config),\n'
                 '    )\n'
                 '    _warn_if_unannotated_eagle_mamba(vllm_config, groups)\n'
                 '    return groups\n'
                 '\n'
                 '\n'
                 'def _is_deepseek_v4_eagle(vllm_config: VllmConfig) -> bool:\n'
                 '    spec_config = vllm_config.speculative_config\n'
                 '    if spec_config is None or not spec_config.use_eagle():\n'
                 '        return False\n'
                 '    model_config = vllm_config.model_config\n'
                 ),
                ('        groups,\n'
                 '        use_trailing_layer_fallback=_uses_trailing_mtp_layers(vllm_config),\n'
                 '    )\n'
                 '    _warn_if_unannotated_eagle_mamba(vllm_config, groups)\n'
                 '    return groups\n'
                 '\n'
                 '\n'
                 'def _uses_trailing_mtp_layers(vllm_config: VllmConfig) -> bool:\n'
                 '    spec_config = vllm_config.speculative_config\n'
                 '    if spec_config is None or not spec_config.use_eagle():\n'
                 '        return False\n'
                 '    if spec_config.method == "mtp":\n'
                 '        return True\n'
                 '    model_config = vllm_config.model_config\n'
                 ),
            ),
            (
                ('\n'
                 'def _annotate_eagle_groups(\n'
                 '    vllm_config: VllmConfig,\n'
                 '    kv_cache_spec: dict[str, KVCacheSpec],\n'
                 '    kv_cache_groups: list[KVCacheGroupSpec],\n'
                 '    use_deepseek_v4_fallback: bool = False,\n'
                 ') -> None:\n'
                 ),
                ('\n'
                 'def _groups_partition_layers_exactly(\n'
                 '    kv_cache_spec: dict[str, KVCacheSpec],\n'
                 '    kv_cache_groups: list[KVCacheGroupSpec],\n'
                 ') -> bool:\n'
                 '    names = [name for group in kv_cache_groups for name in group.layer_names]\n'
                 '    return len(names) == len(kv_cache_spec) and set(names) == set(kv_cache_spec)\n'
                 '\n'
                 '\n'
                 'def _annotate_eagle_groups(\n'
                 '    vllm_config: VllmConfig,\n'
                 '    kv_cache_spec: dict[str, KVCacheSpec],\n'
                 '    kv_cache_groups: list[KVCacheGroupSpec],\n'
                 '    use_trailing_layer_fallback: bool = False,\n'
                 ') -> None:\n'
                 ),
            ),
            (
                ("       target's cannot be found this way.\n"
                 '    2. Model-scoped positional fallback for DeepseekV4/V4.1, whose MTP block\n'
                 "       reuses the target's own decoder layer and so carries no spec marker. Its\n"
                 '       draft attention layer is always the last registered layer, so flag whichever\n'
                 '       group holds it. This rule is only valid where the groups partition\n'
                 '       exactly the layers of ``kv_cache_spec``, which is true on the packed\n'
                 '       grouping path and not in general; other callers must leave\n'
                 '       ``use_deepseek_v4_fallback`` False. The caller gates this fallback on\n'
                 '       the configured model type.\n'
                 '       FIXME(yifan): avoid/generalize this hacky check.\n'
                 ),
                ("       target's cannot be found this way.\n"
                 '    2. Positional fallback for MTP drafters (including DeepseekV4/V4.1 DSpark), whose\n'
                 "       MTP block reuses the target's own decoder layer and so carries no spec\n"
                 '       marker. Their draft attention layers always register after every\n'
                 '       target layer, so flag whichever group holds the last registered layer.\n'
                 '       This rule is only valid where the groups partition exactly the layers\n'
                 '       of ``kv_cache_spec``, which is re-checked here before applying it.\n'
                 "       When a drafter's trailing caches span several groups, this rule flags\n"
                 '       only the group holding the very last layer and must be generalized.\n'
                 '       FIXME(yifan): avoid/generalize this hacky check.\n'
                 ),
            ),
            (
                ('        kv_cache_groups: Groups to annotate in place.\n'
                 '        use_deepseek_v4_fallback: Enable rule 2 for a DeepseekV4/V4.1 packed\n'
                 '            group.\n'
                 '    """\n'
                 ),
                ('        kv_cache_groups: Groups to annotate in place.\n'
                 '        use_trailing_layer_fallback: Enable rule 2. Callers gate this on\n'
                 '            ``_uses_trailing_mtp_layers``.\n'
                 '    """\n'
                 ),
            ),
            (
                ('\n'
                 '    if not use_deepseek_v4_fallback:\n'
                 '        return\n'
                 ),
                ('\n'
                 '    if not use_trailing_layer_fallback:\n'
                 '        return\n'
                 '    if not _groups_partition_layers_exactly(kv_cache_spec, kv_cache_groups):\n'
                 '        return\n'
                 ),
            ),
            (
                ('            groups.append(KVCacheGroupSpec([name], aligned))\n'
                 '\n'
                 '    _annotate_eagle_groups(vllm_config, kv_cache_spec, groups)\n'
                 '    _warn_if_unannotated_eagle_mamba(vllm_config, groups)\n'
                 '    return groups\n'
                 ),
                ('            groups.append(KVCacheGroupSpec([name], aligned))\n'
                 '\n'
                 '    _annotate_eagle_groups(\n'
                 '        vllm_config,\n'
                 '        kv_cache_spec,\n'
                 '        groups,\n'
                 '        use_trailing_layer_fallback=_uses_trailing_mtp_layers(vllm_config),\n'
                 '    )\n'
                 '    _warn_if_unannotated_eagle_mamba(vllm_config, groups)\n'
                 '    return groups\n'
                 ),
            ),
        ),
    ),
    'vllm/v1/core/kv_cache_coordinator.py': (
        'bf342f79f2bc8d6acb69d3becefb334509e84cae888d42bbd421a8dc8ca9eca3',
        (
            (
                ('    kv_cache_config: KVCacheConfig,\n'
                 ') -> None:\n'
                 '    if retention_interval is None:\n'
                 '        return\n'
                 '\n'
                 ),
                ('    kv_cache_config: KVCacheConfig,\n'
                 ') -> int | None:\n'
                 '    if retention_interval is None:\n'
                 '        return None\n'
                 '\n'
                 ),
            ),
            (
                ('    ):\n'
                 '        if retention_interval == 0:\n'
                 '            return\n'
                 '        raise ValueError(\n'
                 '            "prefix_cache_retention_interval is set but this model has "\n'
                 '            "no sliding-window or Mamba KV cache group, so retention has no "\n'
                 '            "effect. Set it to 0 (it only applies to sliding-window and Mamba "\n'
                 '            "attention)."\n'
                 '        )\n'
                 '\n'
                 '    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:\n'
                 '        raise ValueError(\n'
                 '            f"prefix_cache_retention_interval ({retention_interval}) "\n'
                 '            "must be non-negative and a multiple of scheduler_block_size "\n'
                 '            f"({scheduler_block_size})."\n'
                 '        )\n'
                 '\n'
                 ),
                ('    ):\n'
                 '        if retention_interval != 0:\n'
                 '            logger.warning_once(\n'
                 '                "prefix_cache_retention_interval (%d) has no effect: this model "\n'
                 '                "has no sliding-window or Mamba KV cache group.",\n'
                 '                retention_interval,\n'
                 '            )\n'
                 '        return retention_interval\n'
                 '\n'
                 '    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:\n'
                 '        rounded = cdiv(max(retention_interval, 0), scheduler_block_size) * (\n'
                 '            scheduler_block_size\n'
                 '        )\n'
                 '        logger.warning_once(\n'
                 '            "prefix_cache_retention_interval (%d) is not a non-negative multiple "\n'
                 '            "of scheduler_block_size (%d); using %d.",\n'
                 '            retention_interval,\n'
                 '            scheduler_block_size,\n'
                 '            rounded,\n'
                 '        )\n'
                 '        return rounded\n'
                 '    return retention_interval\n'
                 '\n'
                 ),
            ),
            (
                ('        # 0 = keep only the latest replay boundary; None = dense;\n'
                 '        self.retention_interval = kv_cache_config.prefix_cache_retention_interval\n'
                 '        _validate_prefix_cache_retention_interval(\n'
                 '            self.retention_interval, self.scheduler_block_size, kv_cache_config\n'
                 '        )\n'
                 '\n'
                 ),
                ('        # 0 = keep only the latest replay boundary; None = dense;\n'
                 '        self.retention_interval = _validate_prefix_cache_retention_interval(\n'
                 '            kv_cache_config.prefix_cache_retention_interval,\n'
                 '            self.scheduler_block_size,\n'
                 '            kv_cache_config,\n'
                 '        )\n'
                 '        kv_cache_config.prefix_cache_retention_interval = self.retention_interval\n'
                 '\n'
                 ),
            ),
            (
                ('        extension = request.num_prompt_tokens // block * block\n'
                 '        return tuple(sorted({max(resend - block, 0), max(extension - block, 0)}))\n'
                 '\n'
                 ),
                ('        extension = request.num_prompt_tokens // block * block\n'
                 '        boundaries = {max(resend - block, 0), max(extension - block, 0)}\n'
                 '        if self.enable_partial_hash_hits:\n'
                 '            boundaries.add(resend)\n'
                 '        return tuple(sorted(boundaries))\n'
                 '\n'
                 ),
            ),
            (
                ('            curr_hit_length = hit_length\n'
                 '\n'
                 ),
                ('            curr_hit_length = hit_length\n'
                 '            eagle_boundary_established = False\n'
                 '\n'
                 ),
            ),
            (
                ('                    )\n'
                 '                    continue\n'
                 '\n'
                 '                drop_eagle_block = use_eagle and idx not in eagle_verified\n'
                 '\n'
                 '                _max_length = curr_hit_length\n'
                 '                # Eagle matches one extra drop unit (one hash unit for\n'
                 '                # fine-grained managers, else one cache block) and then drops\n'
                 '                # it, landing back at the candidate length. No margin for\n'
                 '                # mamba: its finder never drops (draft models have no mamba\n'
                 '                # layers), so the hit would grow past the candidate.\n'
                 '                if drop_eagle_block and not isinstance(spec, MambaSpec):\n'
                 ),
                ('                    )\n'
                 '                    if use_eagle and idx in eagle_verified:\n'
                 '                        eagle_boundary_established = True\n'
                 '                    continue\n'
                 '\n'
                 '                drop_eagle_block = use_eagle and idx not in eagle_verified\n'
                 '                mamba_reuses_eagle_boundary = (\n'
                 '                    drop_eagle_block\n'
                 '                    and isinstance(spec, MambaSpec)\n'
                 '                    and eagle_boundary_established\n'
                 '                )\n'
                 '                if mamba_reuses_eagle_boundary:\n'
                 '                    drop_eagle_block = False\n'
                 '\n'
                 '                _max_length = curr_hit_length\n'
                 '                # Non-Mamba finders use one extra cache unit as an EAGLE\n'
                 '                # margin, then drop it to land back at the candidate length.\n'
                 '                # Mamba instead drops relative to its actual recurrent-state\n'
                 '                # checkpoint, so a request-wide margin must not be added here.\n'
                 '                if drop_eagle_block and not isinstance(spec, MambaSpec):\n'
                 ),
            ),
            (
                ('                )\n'
                 '                if drop_eagle_block:\n'
                 '                    eagle_verified.add(idx)\n'
                 '                elif _new_hit_length < curr_hit_length:\n'
                 '                    # length shrunk; invalidate previous eagle verifications\n'
                 '                    eagle_verified.clear()\n'
                 '                curr_hit_length = _new_hit_length\n'
                 ),
                ('                )\n'
                 '                if mamba_reuses_eagle_boundary and _new_hit_length < curr_hit_length:\n'
                 '                    hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(\n'
                 '                        block_hashes=block_hashes,\n'
                 '                        max_length=_max_length,\n'
                 '                        kv_cache_group_ids=group_ids,\n'
                 '                        block_pool=self.single_type_managers[first_group_id].block_pool,\n'
                 '                        kv_cache_spec=spec,\n'
                 '                        drop_eagle_block=True,\n'
                 '                        alignment_tokens=self._cache_hit_alignment_tokens,\n'
                 '                        dcp_world_size=self.single_type_managers[\n'
                 '                            first_group_id\n'
                 '                        ].dcp_world_size,\n'
                 '                        pcp_world_size=self.single_type_managers[\n'
                 '                            first_group_id\n'
                 '                        ].pcp_world_size,\n'
                 '                    )\n'
                 '                    drop_eagle_block = True\n'
                 '                    mamba_reuses_eagle_boundary = False\n'
                 '                if drop_eagle_block:\n'
                 '                    eagle_verified.add(idx)\n'
                 '                    eagle_boundary_established = True\n'
                 '                elif _new_hit_length < curr_hit_length:\n'
                 '                    # length shrunk; invalidate previous eagle verifications\n'
                 '                    eagle_verified.clear()\n'
                 '                    eagle_boundary_established = False\n'
                 '                elif mamba_reuses_eagle_boundary:\n'
                 '                    eagle_verified.add(idx)\n'
                 '                curr_hit_length = _new_hit_length\n'
                 ),
            ),
        ),
    ),
    'vllm/v1/core/single_type_kv_cache_manager.py': (
        '0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf',
        (
            (
                ('            self._num_retired_blocks: dict[str, int] = {}\n'
                 '            # The set of the requests that have been allocated blocks\n'
                 ),
                ('            self._num_retired_blocks: dict[str, int] = {}\n'
                 '            self._retained_state_block_idx: dict[str, int] = {}\n'
                 '            self._retained_tail_blocks: dict[str, KVCacheBlock] = {}\n'
                 '            # The set of the requests that have been allocated blocks\n'
                 ),
            ),
            (
                ('        assert pcp_world_size == 1, "PCP not support mamba now."\n'
                 '        block_hashes = resolve_block_hashes(\n'
                 ),
                ('        assert pcp_world_size == 1, "PCP not support mamba now."\n'
                 '        block_size = kv_cache_spec.block_size\n'
                 '        if drop_eagle_block:\n'
                 '            _, matched_length = cls.find_longest_cache_hit(\n'
                 '                block_hashes=block_hashes,\n'
                 '                max_length=max_length,\n'
                 '                kv_cache_group_ids=kv_cache_group_ids,\n'
                 '                block_pool=block_pool,\n'
                 '                kv_cache_spec=kv_cache_spec,\n'
                 '                drop_eagle_block=False,\n'
                 '                alignment_tokens=alignment_tokens,\n'
                 '                dcp_world_size=dcp_world_size,\n'
                 '                pcp_world_size=pcp_world_size,\n'
                 '            )\n'
                 '\n'
                 '            return cls.find_longest_cache_hit(\n'
                 '                block_hashes=block_hashes,\n'
                 '                max_length=max(matched_length - block_size, 0),\n'
                 '                kv_cache_group_ids=kv_cache_group_ids,\n'
                 '                block_pool=block_pool,\n'
                 '                kv_cache_spec=kv_cache_spec,\n'
                 '                drop_eagle_block=False,\n'
                 '                alignment_tokens=alignment_tokens,\n'
                 '                dcp_world_size=dcp_world_size,\n'
                 '                pcp_world_size=pcp_world_size,\n'
                 '            )\n'
                 '\n'
                 '        block_hashes = resolve_block_hashes(\n'
                 ),
            ),
            (
                ('\n'
                 '        block_size = kv_cache_spec.block_size\n'
                 '        if alignment_tokens < block_size and block_size % alignment_tokens == 0:\n'
                 ),
                ('\n'
                 '        if alignment_tokens < block_size and block_size % alignment_tokens == 0:\n'
                 ),
            ),
            (
                ('        for i in range(last_block - 1, first_block - 1, -1):\n'
                 '            if blocks[i].is_null:\n'
                 '                continue\n'
                 '            freed.append(blocks[i])\n'
                 '            blocks[i] = self._null_block\n'
                 '        if freed:\n'
                 '            self.block_pool.free_blocks(freed)\n'
                 '        self._num_retired_blocks[request_id] = last_block\n'
                 '\n'
                 ),
                ('        for i in range(last_block - 1, first_block - 1, -1):\n'
                 '            block = self._release_state_block(request_id, i)\n'
                 '            if block is not None:\n'
                 '                freed.append(block)\n'
                 '        if freed:\n'
                 '            self.block_pool.free_blocks(freed)\n'
                 '        self._num_retired_blocks[request_id] = last_block\n'
                 '\n'
                 '    def _release_state_block(\n'
                 '        self, request_id: str, block_idx: int\n'
                 '    ) -> KVCacheBlock | None:\n'
                 '        blocks = self.req_to_blocks[request_id]\n'
                 '        block = blocks[block_idx]\n'
                 '        if block.is_null:\n'
                 '            return None\n'
                 '        retained_idx = self._retained_state_block_idx.get(request_id)\n'
                 '        if block_idx == retained_idx:\n'
                 '            return None\n'
                 '        if (\n'
                 '            block.block_hash is not None\n'
                 '            and request_id not in self._retained_tail_blocks\n'
                 '            and (retained_idx is None or retained_idx < block_idx)\n'
                 '            and not self._holds_later_boundary_state(blocks, block_idx)\n'
                 '        ):\n'
                 '            self._retained_state_block_idx[request_id] = block_idx\n'
                 '            if retained_idx is None:\n'
                 '                return None\n'
                 '            block_idx, block = retained_idx, blocks[retained_idx]\n'
                 '        blocks[block_idx] = self._null_block\n'
                 '        return block\n'
                 '\n'
                 '    def _holds_later_boundary_state(\n'
                 '        self, blocks: list[KVCacheBlock], block_idx: int\n'
                 '    ) -> bool:\n'
                 '        return any(\n'
                 '            not block.is_null\n'
                 '            and block.block_hash_num_tokens == (idx + 1) * self.block_size\n'
                 '            for idx, block in enumerate(blocks[block_idx + 1 :], start=block_idx + 1)\n'
                 '        )\n'
                 '\n'
                 '    def _retain_tail_copy(self, request_id: str, cow_block: KVCacheBlock) -> None:\n'
                 '        released: list[KVCacheBlock] = []\n'
                 '        previous_copy = self._retained_tail_blocks.get(request_id)\n'
                 '        if previous_copy is not None:\n'
                 '            released.append(previous_copy)\n'
                 '        retained_idx = self._retained_state_block_idx.pop(request_id, None)\n'
                 '        if retained_idx is not None:\n'
                 '            blocks = self.req_to_blocks[request_id]\n'
                 '            released.append(blocks[retained_idx])\n'
                 '            blocks[retained_idx] = self._null_block\n'
                 '        cow_block.ref_cnt += 1\n'
                 '        self._retained_tail_blocks[request_id] = cow_block\n'
                 '        if released:\n'
                 '            self.block_pool.free_blocks(released)\n'
                 '\n'
                 ),
            ),
            (
                ('            ):\n'
                 '                blocks = self.req_to_blocks[request_id]\n'
                 '                if blocks[last_state_block_idx] != self._null_block:\n'
                 '                    self.block_pool.free_blocks([blocks[last_state_block_idx]])\n'
                 '                    blocks[last_state_block_idx] = self._null_block\n'
                 '\n'
                 ),
                ('            ):\n'
                 '                block = self._release_state_block(request_id, last_state_block_idx)\n'
                 '                if block is not None:\n'
                 '                    self.block_pool.free_blocks([block])\n'
                 '\n'
                 ),
            ),
            (
                ('                            )\n'
                 '                        if cow_block.block_hash is not None:\n'
                 ),
                ('                            )\n'
                 '                            self._retain_tail_copy(request_id, cow_block)\n'
                 '                        if cow_block.block_hash is not None:\n'
                 ),
            ),
            (
                ('    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:\n'
                 '        if self.mamba_cache_mode == "align":\n'
                 '            self._allocated_block_reqs.discard(request_id)\n'
                 '            self.last_state_block_idx.pop(request_id, None)\n'
                 '            self._num_retired_blocks.pop(request_id, None)\n'
                 '            self._checkpoints.pop(request_id, None)\n'
                 ),
                ('    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:\n'
                 '        retained_tail = None\n'
                 '        if self.mamba_cache_mode == "align":\n'
                 '            self._allocated_block_reqs.discard(request_id)\n'
                 '            self.last_state_block_idx.pop(request_id, None)\n'
                 '            self._num_retired_blocks.pop(request_id, None)\n'
                 '            self._retained_state_block_idx.pop(request_id, None)\n'
                 '            retained_tail = self._retained_tail_blocks.pop(request_id, None)\n'
                 '            self._checkpoints.pop(request_id, None)\n'
                 ),
            ),
            (
                ('            ]\n'
                 '        return super().pop_blocks_for_free(request_id)\n'
                 '\n'
                 ),
                ('            ]\n'
                 '        blocks = super().pop_blocks_for_free(request_id)\n'
                 '        if retained_tail is None:\n'
                 '            return blocks\n'
                 '        return [retained_tail, *blocks]\n'
                 '\n'
                 ),
            ),
            (
                ('        ) * hash_block_size\n'
                 '        if self.use_eagle:\n'
                 '            # Eagle groups match one hash unit past the candidate and drop it,\n'
                 ),
                ('        ) * hash_block_size\n'
                 '        if self.use_eagle or self.drop_eagle_checkpoint_block:\n'
                 '            # Eagle groups match one hash unit past the candidate and drop it,\n'
                 ),
            ),
        ),
    ),
    'vllm/v1/simple_kv_offload/manager.py': (
        '3cc841ff97e4ec237800d4f81dcb5dfb5191c639f275a32892915e8b1af43eb7',
        (
            (
                ('                continue\n'
                 '            # FIXME (yifan): handle CPU cache eviction, where\n'
                 ),
                ('                continue\n'
                 '            if not group_manager.kv_cache_spec.prefix_cacheable:\n'
                 '                continue\n'
                 '            # FIXME (yifan): handle CPU cache eviction, where\n'
                 ),
            ),
        ),
    ),
    'vllm/platforms/interface.py': (
        'e5e73c668d8710e5f5fe72c3bf5e1d8cb3875ceaa31dbd910f5c92fb4a42355c',
        (
            (
                ('                attn_block_size = indexer_align * cdiv(attn_block_size, indexer_align)\n'
                 ),
                ('                attn_block_size = indexer_align * cdiv(attn_block_size, indexer_align)\n'
                 '            if cache_config.prefix_match_unit:\n'
                 '                match_align = lcm(cache_config.prefix_match_unit, indexer_align or 1)\n'
                 '                attn_block_size = match_align * cdiv(attn_block_size, match_align)\n'
                 ),
            ),
        ),
    ),
}
