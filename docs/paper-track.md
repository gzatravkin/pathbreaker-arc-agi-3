# Let the model tell you where the compute goes

### Throughput is the budget on ARC-AGI-3: a 512-expert agent at 110 sessions on one GPU, allocated by the model's own router, attention and level rate

**Submission:** ARC-AGI-3, public leaderboard 54.55, rank 6 of 5,073. Code, serve patches and the full write-up:
github.com/gzatravkin/pathbreaker-arc-agi-3.

## Abstract

ARC-AGI-3 prices an agent's *actions* quadratically and its *thinking* not at all. Under a fixed budget — one GPU, nine
hours, 110 unseen games — an LLM agent's score is therefore bounded by how many tokens of deliberation the hardware
can produce per game. We show why a decode step is mostly fixed cost, so throughput is set by how many sessions fit in
memory, and we describe how we took one RTX PRO 6000 from ~600 generated tokens/s at 10 sessions to ~3,500 at 110 concurrent
agent sessions without changing the model's behaviour. Three allocation decisions were made by signals the model already
computes: the **router** chose which experts to keep, the **sparse-attention indexer** showed which context tokens the
model actually reads, and the **level rate** told the run when to trade sessions for a better model — a live switch
from 400 to 512 experts without a restart.

## 1. Why throughput is the lever

A level is worth `(human actions / agent actions)^2`, so every wasted action costs quadratically and every token spent
choosing the next action is free. On our serve the decode step is

| concurrent sessions | 30 | 55 | 80 | 104 |
|---|---|---|---|---|
| step, ms | 47.1 | 60.0 | 69.5 | 76.7 |

— a line, **36.5 ms + 0.40 ms per session** (residuals < 1.6 ms). Each session gains ~3 tokens a step, so
throughput ≈ 3n / (36.5 + 0.4n): it is still half its asymptote at 92 sessions. At 10 sessions, 90% of every step is
spent streaming 40 GB of expert weights that the 11th session would get for free. Hence the order of our work:
memory (how many sessions fit), then scheduling (how much prefill is wasted re-reading prompts), then kernels.

But throughput only counts if behaviour is unchanged. Our first doubling — a 320-expert cut ordered by weight
saliency — moved the board by 0.04; the same agent on a clean serve later won 41% more levels at an equal token
budget. Every change since passed a two-part gate: next-token KL against the unmodified model over 537K reply tokens,
as a multiple of the model's own noise floor, and a behavioural replay of recorded sessions.

## 2. Signal one: the router chooses the experts

The model (Qwen3.8-Flash-Next, W4A16) has 512 routed experts per layer, top-10. With all of them, about half as
many sessions fit. We counted the router's real choices on agent traffic. Routing is concentrated — the top 64 experts take
47% of a median layer's picks (uniform: 12.5%), the top 400 take 98.9% — and it is specialized **by kind of token, not
by game**: on a 2.05M-token trace from 25 games, the top-100 experts of two different games overlap at Jaccard 0.66,
those of reasoning tokens and image tokens at 0.26; image tokens use a separate, narrow set (56 effective experts in
the last layer against 148 for reasoning).

| 400-of-512 cut ordered by | KL × noise floor | prompt routing lost | generated routing lost |
|---|---|---|---|
| router usage | **1.50×** | 3.25% | 0.76% |
| weight saliency | 3.54× | 14.3% | 7.5% |

The saliency cut is the one that cost 30% of the levels on another model (61 against 87, p = 0.007). Usage keeps
what the agent routes to; ranked on mixed traffic it halves the prompt-side loss again (1.77%).

## 3. Signal two: the attention indexer shows what the model reads

Twelve of the 48 layers are sparse attention: a light indexer scores the whole context and the layer reads the top
~2,048 KV rows. We exploit that twice. **For memory**, every session's full KV lives in 84 GiB of host RAM and the GPU
keeps a 12,288-row hot region filled by the indexer's picks; misses are copied over PCIe in one flat launch for all
sessions. Sized on 13,824 recorded steps, LRU misses 6.6% of picked rows (clairvoyant Belady: 4.8%). With int4 KV
(3.56M tokens of pool), resumable sessions (92–100% of turns extend the held sequence instead of re-reading it) and
CUDA graphs for mixed steps, 110 sessions run at once.

**As an instrument**, the picks are a free profiler of the agent's context. On a 118K-token request (12 layers,
49 steps, 1.2M picks):

| segment | share of picks | share of prompt |
|---|---|---|
| the model's own reasoning | **51.9%** | 42.2% |
| tool output | **25.0%** | 17.6% |
| tool-call code | 10.5% | 10.4% |
| board images | 5.6% | 12.5% |
| instructions and frame text | 3.8% | 12.7% |

The agent reads its own written account of the game — what it tried and what happened — and the numbers in it
(digit tokens: 25% of picks against 17% of tokens); it consults the board images at under half their share. The last
two messages take 35% of picks at 2% of the prompt, yet 22% of picks reach 64K+ tokens back: the long context is used,
thinly. Early layers read reasoning (75%); later layers shift to tool output and digits. It fits a negative result
we had no explanation for: no change to the image channel ever moved our score. It is one request, and we
present it as a method more than a law — but it costs nothing on any DSA-style sparse-attention model.

## 4. Signal three: the level rate says when to switch from speed to quality

Inside one run the right model changes. Early, many games are easy and each of 110 sessions wins levels; late, the
rate flattens while the GPU still produces 2,000+ tokens/s.

| play minute | 30 | 60 | 120 | 180 | 240 |
|---|---|---|---|---|---|
| 400 experts, 110 sessions | 212 | 334 | 489 | 554 | 582 |
| 512 experts from minute 0, 45 sessions | 122 | 221 | 362 | 468 | 532 |
| 512 experts from minute 0, 60 sessions | 138 | 217 | 326 | 415 | 480 |

(Cumulative levels, 110 copies of the public games.) The full model cannot buy the early levels, and more sessions
make it worse: at 60 its KV pool sits at ~100% and sessions are closed under pressure. So the serve switches models
mid-run. At load it serves 400 experts and builds the 112 cold experts per layer into the kernel format on disk; the
router is already 512 wide, with the cold rows masked. At the switch the harness parks the sessions with the least
game left, the scheduler drains the tail of the KV pool, the 12.8 GiB of cold experts are copied into it in 25–30 s
and unmasked — no restart, 95 s end to end — and 45 sessions continue on the full model. Our best entry before the
switch read 50.48; with the switch at minute 360 it read **54.55**. The trigger can come from the run itself: "levels
in the last 30 minutes below 8% of the rate in minutes 30–60" fires at minute 245 on the rehearsal above.

## 5. What the gate caught

Per-token drift is necessary and not sufficient. An int8 Gated-DeltaNet state sat at 1.01× the noise floor and still
made replies that end inside the reasoning block ~3× as frequent; a faster CUDA verify kernel raised malformed tool
calls from 0.29% to 0.78%; an int6 KV variant broke tool calls outright (2 levels by minute 7 against 50). We rejected
batch-shared experts (+11.7% decode, 2.90× floor) and top-9 routing (2.12×). The bundle that reverted the lossy
recurrent-state and trim settings in the final week read 50.48, against a previous best of 47.19. Our draws are noisy — one unchanged bundle read
21.48 and 26.06 — so attribution happened on recorded sessions, never on the board.

## 6. Universality

Nothing here is specific to ARC: any agent with a time budget on fixed hardware is in the fixed-cost regime of
Section 1, and the three signals exist in any sparse-MoE, sparse-attention model. The lessons: KV residency is the
product for agent workloads; let the model's router, attention and progress allocate memory; and measure quality
where the agent acts, not per token.

**Links:** code, the 250-patch vLLM serve and the long write-up with every measurement: github.com/gzatravkin/pathbreaker-arc-agi-3.
Built on Daniel Franzen's milestone-2 agent, Tufa Labs' Duck harness, Intel AutoRound and vLLM. Team: Grigoriy
Zatravkin, Florencia Solange; deepdreamer and aizealotwu.
