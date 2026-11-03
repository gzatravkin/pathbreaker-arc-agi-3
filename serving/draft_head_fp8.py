import os

import torch

ENV = "TAAF_VLLM_DRAFT_HEAD_FP8"
E4M3_MAX = 448.0
ALIGN = 16
CHUNK = 8192
_said = []


def enabled() -> bool:
    said = str(os.environ.get(ENV) or "").strip()
    if said not in ("", "0", "1"):
        raise RuntimeError(f"{ENV}={said!r} must be 0 or 1")
    return said == "1"


def quantize_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count, width = rows.shape
    padded = count + (-count % ALIGN)
    w8 = torch.zeros((padded, width), dtype=torch.float8_e4m3fn, device=rows.device)
    scale = torch.ones((padded,), dtype=torch.float32, device=rows.device)
    for start in range(0, count, CHUNK):
        part = rows[start:start + CHUNK].float()
        row_scale = (part.abs().amax(dim=1) / E4M3_MAX).clamp_min(1e-12)
        w8[start:start + part.shape[0]] = (part / row_scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(
            torch.float8_e4m3fn)
        scale[start:start + part.shape[0]] = row_scale
    return w8, scale


def quantize_acts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.float().abs().amax(dim=-1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
    q = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _vllm_ops():
    try:
        from vllm import _custom_ops as ops

        return ops if hasattr(ops, "cutlass_scaled_mm") and hasattr(ops, "scaled_fp8_quant") else None
    except Exception:
        return None


class Fp8Head:
    def __init__(self, rows: torch.Tensor):
        self.n, self.k = rows.shape
        self.dtype = torch.bfloat16
        self.device = rows.device
        self.w8, scale = quantize_rows(rows)
        self.scale_b = scale.view(1, -1).contiguous()
        self.w8_t = self.w8.t()
        self.ops = _vllm_ops() if rows.is_cuda else None

    @property
    def shape(self):
        return (self.n, self.k)

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.bfloat16).contiguous()
        if self.ops is not None:
            q, scale_a = self.ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
            out = self.ops.cutlass_scaled_mm(q, self.w8_t, scale_a, self.scale_b.view(-1), torch.bfloat16)
        else:
            q, scale_a = quantize_acts(x)
            if x.is_cuda:
                out = torch._scaled_mm(q, self.w8_t, scale_a=scale_a, scale_b=self.scale_b,
                                       out_dtype=torch.bfloat16)
            else:
                out = ((q.float() * scale_a) @ (self.w8.float() * self.scale_b.view(-1, 1)).t()).to(torch.bfloat16)
        if not _said:
            _said.append(True)
            print(f"{ENV} drafting: {x.shape[0]} row(s) x {self.n} lm_head rows x {self.k} through e4m3 weights "
                  f"with per-row scales and per-token activation scales via "
                  f"{'vllm cutlass_scaled_mm' if self.ops is not None else 'torch'}", flush=True)
        return out if out.shape[1] == self.n else out[:, :self.n]


def make(rows: torch.Tensor) -> Fp8Head:
    head = Fp8Head(rows)
    print(f"{ENV} built: {head.n} of {rows.shape[0]} rows x {head.k} as e4m3 (padded to {head.w8.shape[0]}), "
          f"{head.w8.numel() / 2**20:.0f} MiB against {rows.numel() * rows.element_size() / 2**20:.0f} MiB", flush=True)
    return head


def linear(x: torch.Tensor, rows) -> torch.Tensor:
    if isinstance(rows, Fp8Head):
        return rows.linear(x)
    return torch.nn.functional.linear(x.to(rows.dtype), rows)
