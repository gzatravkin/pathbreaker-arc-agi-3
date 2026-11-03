TARGETS = {
    'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': (
        'b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a',
        (
            (
                ('    from flashinfer.gdn_prefill import (\n'
                 '        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,\n'
                 '    )\n'),
                ('    from flashinfer.gdn_prefill import (\n'
                 '        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,\n'
                 '    )\n'
                 '    from vllm import taaf_gdn_cp_split\n'
                 '\n'
                 '    chunk_gated_delta_rule_fi = taaf_gdn_cp_split.wrap(chunk_gated_delta_rule_fi)\n'),
            ),
        ),
    ),
}
