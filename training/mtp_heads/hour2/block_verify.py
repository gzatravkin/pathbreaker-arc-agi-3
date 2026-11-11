import torch


def residual_mass(weight, p, q_at_p):
    return (weight[..., None] * p - q_at_p).clamp(min=0.0).sum(-1)


def block_tokens(q, p_at_q, p, q_at_p, lam=1.0, samples=64, generator=None):
    steps = len(q)
    count = q[0].shape[0]
    picks = [torch.multinomial(q[k], samples, replacement=True, generator=generator) for k in range(steps)]
    ratio = [p_at_q[k].gather(1, picks[k]) / q[k].gather(1, picks[k]).clamp(min=1e-12) for k in range(steps)]
    weight = torch.ones(count, samples, device=q[0].device)
    keep = []
    above = []
    for k in range(steps):
        weight = torch.minimum(weight * lam * ratio[k], torch.ones_like(weight))
        if k + 1 < steps:
            mass = residual_mass(lam * weight, p[k + 1][:, None, :], q_at_p[k + 1][:, None, :])
            room = mass + 1.0 - weight
            keep.append(torch.where(room > 0, mass / room.clamp(min=1e-12), torch.ones_like(mass)))
        else:
            keep.append(weight)
        above.append((q[k] * (p_at_q[k] > q[k]).float()).sum(-1))
    expect = torch.zeros(count, samples, device=q[0].device)
    tail = torch.ones(count, samples, device=q[0].device)
    for k in range(steps - 1, -1, -1):
        expect = expect + (k + 1) * keep[k] * tail
        tail = tail * (1.0 - keep[k])
    return 1.0 + expect.mean(-1), torch.stack(above, -1)


def token_tokens(q, p_at_q, lam=1.0):
    rates = torch.stack([torch.minimum(q[k], lam * p_at_q[k]).sum(-1).clamp(max=1.0) for k in range(len(q))], -1)
    return 1.0 + rates.cumprod(-1).sum(-1)


def first_token_tv(q, p_at_q, p, q_at_p, match, lam=1.0, samples=32, generator=None):
    steps = len(q)
    count, top = q[0].shape
    first = torch.minimum(lam * p_at_q[0] / q[0].clamp(min=1e-12), torch.ones_like(q[0]))
    weight = first[:, :, None].expand(count, top, samples)
    survive = torch.ones(count, top, samples, device=q[0].device)
    for k in range(steps):
        if k:
            picks = torch.multinomial(q[k], samples, replacement=True, generator=generator)
            ratio = p_at_q[k].gather(1, picks) / q[k].gather(1, picks).clamp(min=1e-12)
            weight = torch.minimum(weight * lam * ratio[:, None, :], torch.ones_like(weight))
        if k + 1 < steps:
            mass = residual_mass(lam * weight, p[k + 1][:, None, None, :], q_at_p[k + 1][:, None, None, :])
            room = mass + 1.0 - weight
            keep = torch.where(room > 0, mass / room.clamp(min=1e-12), torch.ones_like(mass))
        else:
            keep = weight
        survive = survive * (1.0 - keep)
    accept = 1.0 - survive.mean(-1)
    return emitted_tv(q[0] * accept, p_at_q[0], p[0], q_at_p[0], match, lam)


def emitted_tv(kept, p_at_q, p, q_at_p, match, lam=1.0):
    lost = 1.0 - kept.sum(-1, keepdim=True)
    extra = (lam * p - q_at_p).clamp(min=0.0)
    extra = extra / extra.sum(-1, keepdim=True).clamp(min=1e-12)
    matched = match.float()
    on_q = kept + lost * (matched * extra[:, :, None]).sum(1)
    alone = 1.0 - matched.sum(-1).clamp(max=1.0)
    off_q = lost * extra * alone
    return 0.5 * ((on_q - p_at_q).abs().sum(-1) + ((off_q - p) * alone).abs().sum(-1))


def token_tv(q, p_at_q, p, q_at_p, match, lam=1.0):
    kept = torch.minimum(q, lam * p_at_q)
    best = (kept - p_at_q).clamp(min=0.0).sum(-1)
    return best, emitted_tv(kept, p_at_q, p, q_at_p, match, lam)
