TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        None,
        (
            (
                ('from vllm.model_executor.utils import set_weight_attrs\n'),
                ('from vllm import taaf_gdn_q8_pack as _taaf_q8_pack\n'
                 'from vllm.model_executor.utils import set_weight_attrs\n'),
            ),
            (
                ('class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):\n'
                 '    def get_state_shape(\n'
                 '        self,\n'
                 '    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:\n'
                 '        return MambaStateShapeCalculator.gated_delta_net_state_shape(\n'
                 '            self.tp_size,\n'
                 '            self.num_k_heads,\n'
                 '            self.num_v_heads,\n'
                 '            self.head_k_dim,\n'
                 '            self.head_v_dim,\n'
                 '            self.conv_kernel_size,\n'
                 '            self.num_spec,\n'
                 '        )\n'),
                ('class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):\n'
                 '    def get_state_shape(\n'
                 '        self,\n'
                 '    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:\n'
                 '        return _taaf_q8_pack.state_shape(self, MambaStateShapeCalculator.gated_delta_net_state_shape(\n'
                 '            self.tp_size,\n'
                 '            self.num_k_heads,\n'
                 '            self.num_v_heads,\n'
                 '            self.head_k_dim,\n'
                 '            self.head_v_dim,\n'
                 '            self.conv_kernel_size,\n'
                 '            self.num_spec,\n'
                 '        ))\n'
                 '\n'
                 '    def get_kv_cache_spec(self, vllm_config):\n'
                 '        return _taaf_q8_pack.spec(self, super().get_kv_cache_spec(vllm_config))\n'
                 '\n'
                 '    def bind_kv_cache(self, kv_cache):\n'
                 '        super().bind_kv_cache(kv_cache)\n'
                 '        _taaf_q8_pack.bind(self)\n'),
            ),
        ),
    ),
    'vllm/v1/hisparse/layout.py': (
        None,
        (
            (
                ('from vllm.v1.kv_cache_layout import KVCacheLayout\n'),
                ('from vllm import taaf_gdn_q8_pack as _taaf_q8_pack\n'
                 'from vllm.v1.kv_cache_layout import KVCacheLayout\n'),
            ),
            (
                ('    return [mla_group, *get_kv_cache_groups(vllm_config, other_specs)]\n'),
                ('    return [mla_group, *_taaf_q8_pack.pair(get_kv_cache_groups(vllm_config, other_specs))]\n'),
            ),
            (
                ('    size = bytes_per_block * num_blocks\n'
                 '    kv_cache_tensors = _build_hisparse_kv_cache_tensors(\n'),
                ('    size = bytes_per_block * num_blocks\n'
                 '    _taaf_q8_pack.check(device_groups, bytes_per_block, num_blocks)\n'
                 '    kv_cache_tensors = _build_hisparse_kv_cache_tensors(\n'),
            ),
        ),
    ),
}
