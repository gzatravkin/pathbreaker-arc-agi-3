import os

import torch
import triton

ENV = "TAAF_VLLM_MARKS"
ON = str(os.environ.get(ENV) or "").strip() == "1"
_BUF = {}
_SAID = set()


@triton.jit
def _mark_hkv_step_metadata_in(x):
    pass


@triton.jit
def _mark_hkv_step_metadata_out(x):
    pass


@triton.jit
def _mark_paused_owner_in(x):
    pass


@triton.jit
def _mark_paused_owner_out(x):
    pass


@triton.jit
def _mark_ple_grouped_a_in(x):
    pass


@triton.jit
def _mark_ple_grouped_a_out(x):
    pass


@triton.jit
def _mark_ple_grouped_b_in(x):
    pass


@triton.jit
def _mark_ple_grouped_b_out(x):
    pass


@triton.jit
def _mark_qsa_kv8_back_in(x):
    pass


@triton.jit
def _mark_qsa_kv8_back_out(x):
    pass


@triton.jit
def _mark_moe_taaf_owner_in(x):
    pass


@triton.jit
def _mark_moe_taaf_owner_out(x):
    pass


@triton.jit
def _mark_gdn_prepare_in(x):
    pass


@triton.jit
def _mark_gdn_prepare_out(x):
    pass


@triton.jit
def _mark_embed_in(x):
    pass


@triton.jit
def _mark_embed_out(x):
    pass


@triton.jit
def _mark_prepare_in(x):
    pass


@triton.jit
def _mark_prepare_out(x):
    pass


@triton.jit
def _mark_post_forward_in(x):
    pass


@triton.jit
def _mark_post_forward_out(x):
    pass


KERNELS = {
    "hkv_step_metadata": (_mark_hkv_step_metadata_in, _mark_hkv_step_metadata_out),
    "paused_owner": (_mark_paused_owner_in, _mark_paused_owner_out),
    "ple_grouped_a": (_mark_ple_grouped_a_in, _mark_ple_grouped_a_out),
    "ple_grouped_b": (_mark_ple_grouped_b_in, _mark_ple_grouped_b_out),
    "qsa_kv8_back": (_mark_qsa_kv8_back_in, _mark_qsa_kv8_back_out),
    "moe_taaf_owner": (_mark_moe_taaf_owner_in, _mark_moe_taaf_owner_out),
    "gdn_prepare": (_mark_gdn_prepare_in, _mark_gdn_prepare_out),
    "embed": (_mark_embed_in, _mark_embed_out),
    "prepare": (_mark_prepare_in, _mark_prepare_out),
    "post_forward": (_mark_post_forward_in, _mark_post_forward_out),
}


def _compiling():
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def mark(site, phase, device=None):
    if not ON or _compiling():
        return
    pair = KERNELS.get(site)
    if pair is None:
        raise KeyError(f"{ENV}: unknown mark site {site!r}")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    buf = _BUF.get(device)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.int32, device=device)
        _BUF[device] = buf
    pair[1 if phase else 0][(1,)](buf)
    if site not in _SAID:
        _SAID.add(site)
        print(f"MARKS live: {site} brackets its kernels with _mark_{site}_in/_out in the profiler trace", flush=True)


def marked(site):
    def wrap(fn):
        if not ON:
            return fn

        def inner(*args, **kwargs):
            mark(site, 0)
            try:
                return fn(*args, **kwargs)
            finally:
                mark(site, 1)

        inner.__name__ = fn.__name__
        inner.__wrapped__ = fn
        return inner

    return wrap
