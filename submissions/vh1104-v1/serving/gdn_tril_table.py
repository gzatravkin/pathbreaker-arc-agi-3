TARGETS = {
    'vllm/third_party/flash_linear_attention/ops/solve_tril.py': (
        'd1a4ff27623a938825052500afa7e2aa24fadd2fa4c9be6376c9128824f86f82',
        (
            (
                ('        for num_warps in [2, 4, 8]\n'
                 '        for num_stages in [2, 3, 4, 5]\n'
                 '    ],\n'
                 '    key=["H", "BT", "IS_VARLEN"],\n'
                 ')\n'
                 '@triton.jit(do_not_specialize=["T"])\n'
                 'def merge_16x16_to_64x64_inverse_kernel(\n'),
                ('        for num_warps in [1, 2, 4, 8]\n'
                 '        for num_stages in [1, 2, 3, 4, 5]\n'
                 '    ],\n'
                 '    key=["H", "BT", "IS_VARLEN"],\n'
                 ')\n'
                 '@triton.jit(do_not_specialize=["T"])\n'
                 'def merge_16x16_to_64x64_inverse_kernel(\n'),
            ),
        ),
    ),
}
