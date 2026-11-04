import time

import torch

MOE_IDENTITY_EXPERTS = (512, 400)
MOE_IDENTITY_ROWS = (112, 440, 512)
MOE_IDENTITY_PAD_ROWS = 440
MOE_IDENTITY_PAD_EVERY = 7
MOE_IDENTITY_SKEW_ROWS = 440
MOE_IDENTITY_SKEW_HOT = 24
MOE_IDENTITY_SKEW_BIAS = 4.0
MOE_IDENTITY_TAAF_EXPERTS = 400
MOE_IDENTITY_TAAF_ROWS = (1160, 2048)
MOE_IDENTITY_TAAF_PAD_ROWS = 1160
MOE_IDENTITY_TAAF_SKEW_ROWS = 1160


def moe_pack_gptq(q):
    experts, size_k, size_n = q.shape
    q = q.view(experts, size_k // 8, 8, size_n)
    packed = torch.zeros((experts, size_k // 8, size_n), dtype=torch.int64, device=q.device)
    for nibble in range(8):
        packed |= q[:, :, nibble, :].to(torch.int64) << (4 * nibble)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)
    return packed.to(torch.int32)


def moe_dequant(q, scales, group):
    return (q.float() - 8.0) * scales.float().repeat_interleave(group, dim=0)


def moe_route(logits, topk):
    weights, ids = torch.topk(torch.softmax(logits.float(), dim=-1), topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.contiguous(), ids.to(torch.int32).contiguous()


def moe_reference(x, q13, s13, q2, s2, weights, ids, group, experts):
    hidden = x.float()
    out = torch.zeros((x.shape[0], q2.shape[2]), dtype=torch.float32, device=x.device)
    valid = ids < experts
    for expert in torch.unique(ids[valid]).tolist():
        rows, slots = torch.nonzero(ids == expert, as_tuple=True)
        mid = hidden[rows] @ moe_dequant(q13[expert], s13[expert], group)
        half = mid.shape[1] // 2
        act = torch.nn.functional.silu(mid[:, :half]) * mid[:, half:]
        down = act @ moe_dequant(q2[expert], s2[expert], group)
        out.index_add_(0, rows, down * weights[rows, slots].unsqueeze(1))
    return out


def moe_error(got, want, rows):
    if not bool(rows.any()):
        return 0.0, 0.0
    gap = (got.float()[rows] - want[rows]).abs().max()
    scale = want[rows].abs().max().clamp_min(1e-30)
    return float(gap), float(gap / scale)


def moe_taaf_rows(prepared, q13, s13, q2, s2, generator, *, hidden, topk, group, device, run_marlin):
    import moe_taaf

    out = []
    experts = MOE_IDENTITY_TAAF_EXPERTS
    w1, w2, w1_scale, w2_scale = prepared[experts]
    cases = ([(rows, False, False) for rows in MOE_IDENTITY_TAAF_ROWS] + [(MOE_IDENTITY_TAAF_PAD_ROWS, True, False)]
             + [(MOE_IDENTITY_TAAF_SKEW_ROWS, False, True)])
    for rows, padded, skewed in cases:
        began = time.monotonic()
        name = f"moe_taaf E{experts} rows{rows}" + (" padded" if padded else "") + (" skewed" if skewed else "")
        try:
            x = torch.randn((rows, hidden), generator=generator, device=device).to(torch.bfloat16)
            logits = torch.randn((rows, experts), generator=generator, device=device)
            if skewed:
                logits[:, :MOE_IDENTITY_SKEW_HOT] += MOE_IDENTITY_SKEW_BIAS
            weights, ids = moe_route(logits, topk)
            real = torch.ones(rows, dtype=torch.bool, device=device)
            if padded:
                real[::MOE_IDENTITY_PAD_EVERY] = False
                ids = torch.where(real.unsqueeze(1), ids, torch.full_like(ids, experts))
            block = moe_taaf.select_block(rows, experts, topk)
            stock = run_marlin(x, w1, w2, w1_scale, w2_scale, weights, ids, experts)
            ours = moe_taaf.moe_forward(x, w1, w1_scale, w2, w2_scale, weights, ids, block=block, chunks1=1,
                                        chunks2=1, warps=moe_taaf.SERVE_WARPS, stages=moe_taaf.SERVE_STAGES,
                                        order=0, unroll=0)
            if ours.is_cuda:
                torch.cuda.synchronize()
            want = moe_reference(x, q13, s13, q2, s2, weights, ids, group, experts)
            _, taaf = moe_error(ours, want, real)
            _, marlin = moe_error(stock, want, real)
            _, gap = moe_error(ours, stock.float(), real)
            finite = bool(torch.isfinite(ours.float()[real]).all())
            verdict = "nonfinite" if not finite else ("taaf-worse" if taaf - marlin > marlin else "taaf-level")
            out.append((name, verdict, f"max rel err {taaf:.3g} for moe_taaf against {marlin:.3g} for Marlin vs the "
                                       f"fp32 reference over {int(real.sum())} real rows; moe_taaf vs Marlin {gap:.3g}; "
                                       f"block {block}", time.monotonic() - began))
        except Exception as error:
            out.append((name, "stopped", f"{type(error).__name__}: {error}"[:300], time.monotonic() - began))
    return out


def moe_identity_check(*, hidden, intermediate, topk, group, device="cuda", seed=3):
    from types import SimpleNamespace

    rows_out = []
    errors = {}
    started = time.monotonic()
    try:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _process_weights_marlin
        from vllm.scalar_type import scalar_types

        def run_marlin(x, w1, w2, w1_scale, w2_scale, weights, ids, experts):
            return fused_marlin_moe(x, w1, w2, None, None, w1_scale, w2_scale, weights, ids, scalar_types.uint4b8.id,
                                    global_num_experts=experts)

        generator = torch.Generator(device=device).manual_seed(seed)
        most = max(MOE_IDENTITY_EXPERTS)
        q13 = torch.randint(0, 16, (most, hidden, 2 * intermediate), generator=generator, device=device,
                            dtype=torch.uint8)
        q2 = torch.randint(0, 16, (most, intermediate, hidden), generator=generator, device=device, dtype=torch.uint8)
        s13 = (0.002 + 0.02 * torch.rand((most, hidden // group, 2 * intermediate), generator=generator,
                                         device=device)).to(torch.bfloat16)
        s2 = (0.002 + 0.02 * torch.rand((most, intermediate // group, hidden), generator=generator,
                                        device=device)).to(torch.bfloat16)
        packed13, packed2 = moe_pack_gptq(q13), moe_pack_gptq(q2)
        layer = SimpleNamespace(intermediate_size_per_partition=intermediate, num_groups_w13=hidden // group,
                                num_groups_w2=intermediate // group)
        prepared = {}
        for experts in MOE_IDENTITY_EXPERTS:
            converted = _process_weights_marlin(layer, None, 4, 8, group, packed13[:experts].contiguous(),
                                                packed2[:experts].contiguous(), s13[:experts].contiguous(),
                                                s2[:experts].contiguous())
            prepared[experts] = converted[:4]
        torch.cuda.synchronize()
        rows_out.append(("moe setup", "ready", f"W4A16 g{group} uint4b8, hidden {hidden}, intermediate "
                                               f"{intermediate}, top-{topk}, {most} synthetic experts repacked for "
                                               f"Marlin at {', '.join(str(e) for e in MOE_IDENTITY_EXPERTS)}",
                         time.monotonic() - started))
        cases = ([(rows, False, False) for rows in MOE_IDENTITY_ROWS] + [(MOE_IDENTITY_PAD_ROWS, True, False)]
                 + [(MOE_IDENTITY_SKEW_ROWS, False, True)])
        for rows, padded, skewed in cases:
            x = (torch.randn((rows, hidden), generator=generator, device=device)).to(torch.bfloat16)
            logits = torch.randn((rows, most), generator=generator, device=device)
            if skewed:
                logits[:, :MOE_IDENTITY_SKEW_HOT] += MOE_IDENTITY_SKEW_BIAS
            for experts in MOE_IDENTITY_EXPERTS:
                began = time.monotonic()
                name = f"moe E{experts} rows{rows}" + (" padded" if padded else "") + (" skewed" if skewed else "")
                try:
                    weights, ids = moe_route(logits[:, :experts], topk)
                    real = torch.ones(rows, dtype=torch.bool, device=device)
                    if padded:
                        real[::MOE_IDENTITY_PAD_EVERY] = False
                        ids = torch.where(real.unsqueeze(1), ids, torch.full_like(ids, experts))
                    w1, w2, w1_scale, w2_scale = prepared[experts]
                    got = run_marlin(x, w1, w2, w1_scale, w2_scale, weights, ids, experts)
                    torch.cuda.synchronize()
                    want = moe_reference(x, q13, s13, q2, s2, weights, ids, group, experts)
                    gap, rel = moe_error(got, want, real)
                    finite = bool(torch.isfinite(got.float()[real]).all())
                    errors[(experts, rows, padded, skewed)] = rel
                    rows_out.append((name, "measured" if finite else "nonfinite",
                                     f"max abs err {gap:.3g}, max rel err {rel:.3g} vs the fp32 reference over "
                                     f"{int(real.sum())} real rows" + (f" ({rows - int(real.sum())} rows carry "
                                                                       f"expert id {experts})" if padded else ""),
                                     time.monotonic() - began))
                except Exception as error:
                    rows_out.append((name, "stopped", f"{type(error).__name__}: {error}"[:300],
                                     time.monotonic() - began))
        lo, hi = min(MOE_IDENTITY_EXPERTS), max(MOE_IDENTITY_EXPERTS)
        for rows, padded, skewed in cases:
            small, large = errors.get((lo, rows, padded, skewed)), errors.get((hi, rows, padded, skewed))
            if small is None or large is None:
                continue
            worse = small - large > large
            rows_out.append((f"moe E{lo} vs E{hi} rows{rows}" + (" padded" if padded else "") + (" skewed" if skewed else ""),
                             f"E{lo}-worse" if worse else f"E{lo}-level",
                             f"max rel err {small:.3g} at E{lo} against {large:.3g} at E{hi}", 0.0))
        if MOE_IDENTITY_TAAF_EXPERTS in prepared:
            try:
                rows_out.extend(moe_taaf_rows(prepared, q13, s13, q2, s2, generator, hidden=hidden, topk=topk,
                                              group=group, device=device, run_marlin=run_marlin))
            except Exception as error:
                rows_out.append(("moe_taaf", "stopped", f"{type(error).__name__}: {error}"[:300], 0.0))
    except Exception as error:
        rows_out.append(("moe", "stopped", f"{type(error).__name__}: {error}"[:300], time.monotonic() - started))
    return rows_out
