TARGETS = {
    'vllm/model_executor/layers/mamba/abstract.py': (
        'dfdc8c836894f574e41d93d76ab989cacd3d7ecf59c97a929abfa48d9364e8ad',
        (
            (
                ('from vllm.config import VllmConfig\n'),
                ('from vllm import taaf_gdn_accept\n'
                 'from vllm.config import VllmConfig\n'),
            ),
            (
                ('                if vllm_config.cache_config.use_kda_recoverssm\n'
                 '                else vllm_config.num_speculative_tokens\n'),
                ('                if vllm_config.cache_config.use_kda_recoverssm\n'
                 '                or taaf_gdn_accept.active(vllm_config)\n'
                 '                else vllm_config.num_speculative_tokens\n'),
            ),
        ),
    ),
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            (
                ('from vllm import envs\n'),
                ('from vllm import envs, taaf_gdn_accept\n'),
            ),
            (
                ('        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)\n'),
                ('        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)\n'
                 '        self.gdn_accept_records = taaf_gdn_accept.allocate_records(self, vllm_config)\n'),
            ),
            (
                ('                max_query_len=spec_state_indices_tensor.size(-1),\n'),
                ('                max_query_len=self.num_spec + 1,\n'),
            ),
            (
                ('        # 2.1: Process the multi-query part\n'
                 '        if spec_sequence_masks is not None:\n'
                 '            core_attn_out_spec, last_recurrent_state = (\n'),
                ('        # 2.1: Process the multi-query part\n'
                 '        if spec_sequence_masks is not None and self.gdn_accept_records is not None:\n'
                 '            core_attn_out_spec = taaf_gdn_accept.verify_layer(\n'
                 '                self,\n'
                 '                query_spec[0],\n'
                 '                key_spec[0],\n'
                 '                value_spec[0],\n'
                 '                a_spec,\n'
                 '                b_spec,\n'
                 '                spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],\n'
                 '                spec_state_indices_tensor[: attn_metadata.num_spec_decodes, 0],\n'
                 '            )\n'
                 '            last_recurrent_state = None\n'
                 '        elif spec_sequence_masks is not None:\n'
                 '            core_attn_out_spec, last_recurrent_state = (\n'),
            ),
            (
                ('            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")\n'
                 '        )\n'),
                ('            and hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")\n'
                 '            and self.gdn_accept_records is None\n'
                 '        )\n'),
            ),
            (
                ('        assert isinstance(attn_metadata, GDNAttentionMetadata)\n'
                 '        if (\n'
                 '            self._can_use_fused_gdn_mtp_decode(attn_metadata)\n'
                 '            and attn_metadata.num_prefills == 0\n'
                 '        ):\n'),
                ('        assert isinstance(attn_metadata, GDNAttentionMetadata)\n'
                 '        if (\n'
                 '            self.gdn_accept_records is not None\n'
                 '            and attn_metadata.spec_sequence_masks is not None\n'
                 '            and attn_metadata.num_prefills == 0\n'
                 '            and attn_metadata.num_decodes == 0\n'
                 '        ):\n'
                 '            taaf_gdn_accept.decode(\n'
                 '                self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata\n'
                 '            )\n'
                 '            return\n'
                 '        if (\n'
                 '            self._can_use_fused_gdn_mtp_decode(attn_metadata)\n'
                 '            and attn_metadata.num_prefills == 0\n'
                 '        ):\n'),
            ),
        ),
    ),
    'vllm/v1/attention/backends/gdn_attn.py': (
        '3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054',
        (
            (
                ('from vllm.config import VllmConfig\n'),
                ('from vllm import taaf_gdn_accept\n'
                 'from vllm.config import VllmConfig\n'),
            ),
            (
                ('from vllm.v1.attention.backends.utils import (\n'),
                ('from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata\n'
                 'from vllm.v1.attention.backends.utils import (\n'),
            ),
            (
                ('    token_chunk_offset_ptr: torch.Tensor | None = None\n'
                 '\n'
                 '\n'
                 'class GDNAttentionMetadataBuilder('),
                ('    token_chunk_offset_ptr: torch.Tensor | None = None\n'
                 '\n'
                 '\n'
                 '@dataclass\n'
                 'class GDNAcceptMetadata(GDNAttentionMetadata, RecoverSSMMetadata):\n'
                 '    accept_commit: object = None\n'
                 '\n'
                 '    def commit_recoverssm_state(self, num_accepted_tokens):\n'
                 '        if self.accept_commit is not None:\n'
                 '            self.accept_commit(num_accepted_tokens)\n'
                 '        return None\n'
                 '\n'
                 '\n'
                 'class GDNAttentionMetadataBuilder('),
            ),
            (
                ('        self._init_reorder_batch_threshold(1, self.use_spec_decode)\n'),
                ('        self._init_reorder_batch_threshold(1, self.use_spec_decode)\n'
                 '        self.gdn_accept = self.use_spec_decode and taaf_gdn_accept.active(vllm_config)\n'
                 '        self.spec_state_slots = 1 if self.gdn_accept else self.num_spec + 1\n'
                 '        self.gdn_accept_context = None\n'
                 '        if self.gdn_accept:\n'
                 '            taaf_gdn_accept.say_group(layer_names, self.num_spec)\n'),
            ),
            (
                ('            (self.decode_cudagraph_max_bs, self.num_spec + 1),\n'),
                ('            (self.decode_cudagraph_max_bs, self.spec_state_slots),\n'),
            ),
            (
                ('            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0\n'
                 '            num_spec_decodes = spec_sequence_masks_cpu.sum().item()\n'
                 '            if (\n'
                 '                num_spec_decodes == 0\n'
                 '                or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()\n'
                 '                == 0\n'
                 '            ):\n'),
                ('            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0\n'
                 '            if self.gdn_accept:\n'
                 '                spec_sequence_masks_cpu = taaf_gdn_accept.decode_rows(\n'
                 '                    spec_sequence_masks_cpu, m, self.num_spec\n'
                 '                )\n'
                 '            num_spec_decodes = spec_sequence_masks_cpu.sum().item()\n'
                 '            if num_spec_decodes == 0 or (\n'
                 '                not self.gdn_accept\n'
                 '                and num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()\n'
                 '                == 0\n'
                 '            ):\n'),
            ),
            (
                ('                # Filter by spec_sequence_masks to exclude padded sequences\n'
                 '                spec_state_indices_tensor = block_table_tensor[\n'
                 '                    spec_sequence_masks_cpu, : self.num_spec + 1\n'
                 '                ]\n'),
                ('                # Filter by spec_sequence_masks to exclude padded sequences\n'
                 '                spec_state_indices_tensor = block_table_tensor[\n'
                 '                    spec_sequence_masks_cpu, : self.spec_state_slots\n'
                 '                ]\n'),
            ),
            (
                ('                spec_state_indices_tensor = block_table_tensor[\n'
                 '                    spec_sequence_masks_cpu, : self.num_spec + 1\n'
                 '                ]\n'
                 '                non_spec_state_indices_tensor = block_table_tensor[\n'),
                ('                spec_state_indices_tensor = block_table_tensor[\n'
                 '                    spec_sequence_masks_cpu, : self.spec_state_slots\n'
                 '                ]\n'
                 '                non_spec_state_indices_tensor = block_table_tensor[\n'),
            ),
            (
                ('        attn_metadata = GDNAttentionMetadata(\n'),
                ('        attn_metadata = (\n'
                 '            GDNAcceptMetadata if self.gdn_accept else GDNAttentionMetadata\n'
                 '        )(\n'),
            ),
            (
                ('            token_chunk_offset_ptr=token_chunk_offset_ptr,\n'
                 '        )\n'
                 '        return attn_metadata\n'),
                ('            token_chunk_offset_ptr=token_chunk_offset_ptr,\n'
                 '        )\n'
                 '        if self.gdn_accept and num_spec_decodes > 0:\n'
                 '            attn_metadata.accept_commit = taaf_gdn_accept.commit_for(\n'
                 '                self,\n'
                 '                m,\n'
                 '                spec_state_indices_tensor,\n'
                 '                spec_query_start_loc,\n'
                 '                num_spec_decodes,\n'
                 '                None\n'
                 '                if num_prefills == 0 and num_decodes == 0\n'
                 '                else async_tensor_h2d(\n'
                 '                    spec_sequence_masks_cpu.nonzero(as_tuple=True)[0],\n'
                 '                    device=query_start_loc.device,\n'
                 '                    dtype=torch.int32,\n'
                 '                ),\n'
                 '            )\n'
                 '        return attn_metadata\n'),
            ),
            (
                ('            chunk_indices, chunk_offsets = self._build_chunk_metadata(\n'
                 '                prefill_query_start_loc,\n'
                 '                prefill_query_start_loc_cpu,\n'
                 '                query_start_loc.device,\n'
                 '            )\n'),
                ('            chunk_indices, chunk_offsets = taaf_gdn_accept.step_memo(\n'
                 '                "chunk",\n'
                 '                (prefill_query_start_loc_cpu,),\n'
                 '                lambda: self._build_chunk_metadata(\n'
                 '                    prefill_query_start_loc,\n'
                 '                    prefill_query_start_loc_cpu,\n'
                 '                    query_start_loc.device,\n'
                 '                ),\n'
                 '            )\n'),
            ),
            (
                ('            nums_dict, batch_ptr, token_chunk_offset_ptr = (\n'
                 '                compute_causal_conv1d_metadata(\n'
                 '                    non_spec_query_start_loc_cpu,\n'
                 '                    device=query_start_loc.device,\n'
                 '                )\n'
                 '            )\n'),
                ('            nums_dict, batch_ptr, token_chunk_offset_ptr = taaf_gdn_accept.step_memo(\n'
                 '                "conv",\n'
                 '                (non_spec_query_start_loc_cpu,),\n'
                 '                lambda: compute_causal_conv1d_metadata(\n'
                 '                    non_spec_query_start_loc_cpu,\n'
                 '                    device=query_start_loc.device,\n'
                 '                ),\n'
                 '            )\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/model_states/mamba_hybrid.py': (
        '573e130ac5587d4822fabf97adad1ad0fda02ff83c5e70df879a5bb175ae845c',
        (
            (
                ('                is_decode = (\n'
                 '                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1\n'
                 '                )\n'),
                ('                is_decode = (\n'
                 '                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1\n'
                 '                ) | ~input_batch.is_prefilling_np\n'),
            ),
            (
                ('from vllm.config import VllmConfig\n'),
                ('from vllm import taaf_gdn_accept\n'
                 'from vllm.config import VllmConfig\n'),
            ),
            (
                ('        self.recoverssm = (\n'
                 '            RecoverSSMState() if self.cache_config.use_kda_recoverssm else None\n'
                 '        )\n'),
                ('        self.gdn_accept = taaf_gdn_accept.active(vllm_config)\n'
                 '        self.recoverssm = (\n'
                 '            RecoverSSMState()\n'
                 '            if self.cache_config.use_kda_recoverssm or self.gdn_accept\n'
                 '            else None\n'
                 '        )\n'),
            ),
            (
                ('                make_buffer=lambda n, dtype: CpuGpuBuffer(\n'
                 '                    n, dtype=dtype, device=self.device\n'
                 '                ),\n'
                 '            )\n'),
                ('                make_buffer=lambda n, dtype: CpuGpuBuffer(\n'
                 '                    n, dtype=dtype, device=self.device\n'
                 '                ),\n'
                 '                recovered_temporal_states=self.gdn_accept,\n'
                 '            )\n'),
            ),
            (
                ('        self.num_accepted_tokens_gpu[req_index].fill_(1)\n'),
                ('        taaf_gdn_accept.unit_resume(self, req_index, new_req_data)\n'
                 '        self.num_accepted_tokens_gpu[req_index].fill_(1)\n'
                 '        taaf_gdn_accept.unit_reset(self, req_index)\n'),
            ),
            (
                ('    def _get_mamba_group_info(\n'),
                ('    def remove_request(self, req_id: str) -> None:\n'
                 '        taaf_gdn_accept.unit_leave(self, req_id)\n'
                 '        super().remove_request(req_id)\n'
                 '\n'
                 '    def _get_mamba_group_info(\n'),
            ),
            (
                ('        mamba_attn_metadata = MambaHybridAttnMetadata(\n'),
                ('        taaf_gdn_accept.unit_prepare(self, input_batch, attn_groups, num_reqs, '
                 'for_capture)\n'
                 '        mamba_attn_metadata = MambaHybridAttnMetadata(\n'),
            ),
            (
                ('        num_reqs = idx_mapping.shape[0]\n'),
                ('        num_reqs = idx_mapping.shape[0]\n'
                 '        taaf_gdn_accept.unit_snapshot(self, idx_mapping, num_sampled, '
                 'num_computed_tokens)\n'),
            ),
        ),
    ),
    'vllm/v1/worker/mamba_utils.py': (
        '23fdaa4e8f63ee8f938eaa1a2b39f803cb04fa6ecd69789563254b1a0691413a',
        (
            (
                ('    TEMPORAL_TILES: tl.constexpr,\n'
                 '):\n'
                 '    """Copy one (layer, state-type) mamba state block between block columns.\n'),
                ('    TEMPORAL_TILES: tl.constexpr,\n'
                 '    TEMPORAL_MODE: tl.constexpr,\n'
                 '):\n'
                 '    """Copy one (layer, state-type) mamba state block between block columns.\n'),
            ),
            (
                ('    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)\n'),
                ('    if TEMPORAL_MODE == 2:\n'
                 '        return\n'
                 '    temporal_col = src_col + token_bias\n'
                 '    if TEMPORAL_MODE == 1:\n'
                 '        temporal_col = src_col\n'
                 '    actual_src_block_id = tl.load(block_table_base + temporal_col).to(tl.int64)\n'),
            ),
            (
                ('    TEMPORAL_TILES: tl.constexpr = 1,\n'
                 '):\n'
                 '    """\n'
                 '    Fused GPU kernel for postprocess_mamba'),
                ('    TEMPORAL_TILES: tl.constexpr = 1,\n'
                 '    TEMPORAL_MODE: tl.constexpr = 0,\n'
                 '):\n'
                 '    """\n'
                 '    Fused GPU kernel for postprocess_mamba'),
            ),
            (
                ('        accept_token_bias,\n'
                 '        block_table_ptrs_ptr,\n'
                 '        block_table_stride_req,\n'
                 '        state_base_addrs_ptr,\n'
                 '        state_block_strides_ptr,\n'
                 '        state_elem_sizes_ptr,\n'
                 '        state_inner_sizes_ptr,\n'
                 '        state_conv_widths_ptr,\n'
                 '        state_group_indices_ptr,\n'
                 '        state_dim_row_count_ptr,\n'
                 '        state_dim_row_stride_ptr,\n'
                 '        tile_idx,\n'
                 '        COPY_BLOCK_SIZE,\n'
                 '        CONV_STATE_DIM_FIRST,\n'
                 '        TEMPORAL_TILES,\n'
                 '    )\n'),
                ('        accept_token_bias,\n'
                 '        block_table_ptrs_ptr,\n'
                 '        block_table_stride_req,\n'
                 '        state_base_addrs_ptr,\n'
                 '        state_block_strides_ptr,\n'
                 '        state_elem_sizes_ptr,\n'
                 '        state_inner_sizes_ptr,\n'
                 '        state_conv_widths_ptr,\n'
                 '        state_group_indices_ptr,\n'
                 '        state_dim_row_count_ptr,\n'
                 '        state_dim_row_stride_ptr,\n'
                 '        tile_idx,\n'
                 '        COPY_BLOCK_SIZE,\n'
                 '        CONV_STATE_DIM_FIRST,\n'
                 '        TEMPORAL_TILES,\n'
                 '        TEMPORAL_MODE,\n'
                 '    )\n'),
            ),
            (
                ('    TEMPORAL_TILES: tl.constexpr = 1,\n'
                 '):\n'
                 '    """Pre-copy mamba "align" state across block boundaries.\n'),
                ('    TEMPORAL_TILES: tl.constexpr = 1,\n'
                 '    TEMPORAL_MODE: tl.constexpr = 0,\n'
                 '):\n'
                 '    """Pre-copy mamba "align" state across block boundaries.\n'),
            ),
            (
                ('        dst_col,\n'
                 '        token_bias,\n'
                 '        block_table_ptrs_ptr,\n'
                 '        block_table_stride_req,\n'
                 '        state_base_addrs_ptr,\n'
                 '        state_block_strides_ptr,\n'
                 '        state_elem_sizes_ptr,\n'
                 '        state_inner_sizes_ptr,\n'
                 '        state_conv_widths_ptr,\n'
                 '        state_group_indices_ptr,\n'
                 '        state_dim_row_count_ptr,\n'
                 '        state_dim_row_stride_ptr,\n'
                 '        tile_idx,\n'
                 '        COPY_BLOCK_SIZE,\n'
                 '        CONV_STATE_DIM_FIRST,\n'
                 '        TEMPORAL_TILES,\n'
                 '    )\n'),
                ('        dst_col,\n'
                 '        token_bias,\n'
                 '        block_table_ptrs_ptr,\n'
                 '        block_table_stride_req,\n'
                 '        state_base_addrs_ptr,\n'
                 '        state_block_strides_ptr,\n'
                 '        state_elem_sizes_ptr,\n'
                 '        state_inner_sizes_ptr,\n'
                 '        state_conv_widths_ptr,\n'
                 '        state_group_indices_ptr,\n'
                 '        state_dim_row_count_ptr,\n'
                 '        state_dim_row_stride_ptr,\n'
                 '        tile_idx,\n'
                 '        COPY_BLOCK_SIZE,\n'
                 '        CONV_STATE_DIM_FIRST,\n'
                 '        TEMPORAL_TILES,\n'
                 '        TEMPORAL_MODE,\n'
                 '    )\n'),
            ),
            (
                ('    # Flag to track if metadata has been populated\n'
                 '    is_initialized: bool = False\n'),
                ('    # Flag to track if metadata has been populated\n'
                 '    is_initialized: bool = False\n'
                 '    recovered_temporal_states: bool = False\n'),
            ),
            (
                ('        make_buffer: Callable[..., CpuGpuBuffer],\n'
                 '    ) -> "MambaSpecDecodeGPUContext":\n'),
                ('        make_buffer: Callable[..., CpuGpuBuffer],\n'
                 '        recovered_temporal_states: bool = False,\n'
                 '    ) -> "MambaSpecDecodeGPUContext":\n'),
            ),
            (
                ('            is_initialized=False,\n'
                 '        )\n'),
                ('            is_initialized=False,\n'
                 '            recovered_temporal_states=recovered_temporal_states,\n'
                 '        )\n'),
            ),
            (
                ('            HAS_IDX_MAPPING=idx_mapping is not None,\n'
                 '            TEMPORAL_TILES=_TEMPORAL_TILES,\n'
                 '        )\n'),
                ('            HAS_IDX_MAPPING=idx_mapping is not None,\n'
                 '            TEMPORAL_TILES=_TEMPORAL_TILES,\n'
                 '            TEMPORAL_MODE=1 if self.recovered_temporal_states else 0,\n'
                 '        )\n'),
            ),
            (
                ('            PRECOMPUTED_NEW_COMPUTED=True,\n'
                 '            TEMPORAL_TILES=_TEMPORAL_TILES,\n'
                 '        )\n'),
                ('            PRECOMPUTED_NEW_COMPUTED=True,\n'
                 '            TEMPORAL_TILES=_TEMPORAL_TILES,\n'
                 '            TEMPORAL_MODE=2 if self.recovered_temporal_states else 0,\n'
                 '        )\n'),
            ),
        ),
    ),
}
