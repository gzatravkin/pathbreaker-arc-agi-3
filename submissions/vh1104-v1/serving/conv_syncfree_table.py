TARGETS = {
    'vllm/v1/attention/backends/short_conv_attn.py': (
        '84fc460ef2fbd09743b069bec0c14b39542af6c2254a403e37a048427e151121',
        (
            (
                ('            token_group = torch.repeat_interleave(req_group, query_lens)\n'),
                ('            from vllm import taaf_conv_syncfree\n'
                 '\n'
                 '            token_group = taaf_conv_syncfree.token_group(req_group, query_lens, '
                 'query_lens_cpu)\n'),
            ),
        ),
    ),
}
