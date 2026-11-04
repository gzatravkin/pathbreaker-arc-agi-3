TARGETS = {
    'vllm/models/qwen4_exp/nvidia/mtp.py': (
        None,
        (
            ('from vllm.model_executor.layers.layernorm import GemmaRMSNorm\n',
             'from vllm.model_executor.layers.layernorm import GemmaRMSNorm\nfrom vllm import taaf_small_kernels as _taaf_small_kernels\n\n_taaf_small_kernels.install_gemma_norm(GemmaRMSNorm)\n'),
        ),
    ),
}
