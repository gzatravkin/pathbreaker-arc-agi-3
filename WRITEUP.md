# Let the model tell you where the compute goes

### A 512-expert LLM agent at 110 concurrent sessions on one GPU for ARC-AGI-3 — public leaderboard rank 6 of 5,073 (54.55)

Grigoriy Zatravkin, with Florencia Solange; deepdreamer and aizealotwu on fine-tuning. Built on Daniel Franzen's
published milestone-2 agent and Tufa Labs' Duck harness.

---

## TL;DR

- **ARC-AGI-3 scores actions, not tokens.** A level is worth `min(115, (human/agent actions)^2 * 100)`, weighted by
  depth; the tokens the agent spends thinking before each action cost nothing on the score. Under a fixed budget —
  one RTX PRO 6000, nine hours, 110 hidden games — the score is bounded by how many tokens of deliberation the
  hardware can produce per game. **Throughput is the budget.**
- **A decode step is mostly fixed cost.** On our serve a step costs ~36.5 ms plus ~0.40 ms per concurrent session
  (four lane counts, fit residuals under 1.6 ms), so aggregate throughput keeps growing almost linearly until ~90
  sessions. The problem was a memory problem first (how many sessions fit), a scheduling problem second (how much
  of the GPU is spent re-reading prompts), and a kernel problem last.
- **We took one GPU from ~600 generated tokens/s at 10 sessions (the published milestone-2 serve) to ~3,500 at 110
  sessions** on agent traffic, by fitting 110 sessions into it: a
  400-of-512 expert cut ordered by the router's own usage, int4 KV, a host-RAM KV tier whose GPU hot region is
  filled by the sparse-attention indexer's own picks, resumable sessions, CUDA graphs for mixed steps, a tuned MTP
  drafter and a dozen SM120 kernels.
- **Speed bought with quality is worth nothing.** The first time throughput doubled, the board moved by 0.04. Every
  later gain had to pass a quality gate — KL against the unmodified model over 537K reply tokens, compared with the
  model's own noise floor — and then a behavioural check, because a per-token metric passed a change that broke whole
  replies. Reverting the lossy settings in the last week took the board from ~40 to 50.48.
- **Speed early, quality late, inside one run.** Early in a run many games are easy and every session pays; late,
  most sessions are stuck on hard levels and the level rate flattens. We built a **live expert switch**: the serve
  starts on the 400-expert cut with 110 sessions, and when the run reaches its late phase it drains the tail of its own
  KV pool, copies the 112 cold experts per layer into that memory and unmasks them in the router — without a restart —
  then plays on with 45 sessions on the full 512-expert model. Our best entry before the switch read 50.48; the entry
  with it read **54.55**, our final public score.
- **The model tells you what it reads.** The sparse-attention indexer picks ~2,048 KV rows per layer per decode step;
  logging those picks costs nothing and shows which parts of a 118K-token agent prompt the model actually consults.
  It reads its own reasoning and the tool outputs, and numbers in them; it reads the board images at less than half
  and the instructions at less than a third of their share of the prompt. The router is the same kind of signal for
  experts. Both decided where our memory went.

---

## 1. The setting

| | |
|---|---|
| Hardware | one NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120), ~180 GB host RAM, no network |
| Budget | 540 minutes of wall clock for 110 hidden games; one scored submission a day |
| Model | Qwen3.8-Flash-Next, Intel AutoRound W4A16: 48 layers, 36 Gated-DeltaNet (linear attention with a recurrent state) and 12 sparse full-attention layers; 512 routed experts per MoE layer, top-10 routing; an MTP draft head |
| Agent | Daniel Franzen's published milestone-2 agent (a tool-using reasoning agent with board images, frame diffs and a Python sandbox), with our changes |
| Serve | vLLM 0.30 with ~250 patches of ours, all switchable by environment knobs and logged when they fire |

Each game is one long conversation: the agent reads a rendering of the board and its history, thinks, calls tools,
acts, and the next turn appends a few hundred tokens to a context that grows to 100K+ tokens. 110 such
conversations run at once, one per game.

## 2. Why throughput is the lever (the theory)

**The score does not price compute.** ARC-AGI-3 measures efficiency in *actions*: a level is worth
`(human actions / agent actions)^2`, so every wasted action is punished quadratically, while every token spent
deciding on the next action is free. For an LLM agent that means more deliberation per action is pure gain until
the model stops improving its decisions. Within a fixed nine-hour budget the only way to buy more deliberation is to
produce more tokens per second.

**A decode step is mostly fixed cost.** Measured on our serve (soak load, full CUDA graphs, four drafted tokens):

| concurrent sessions | 30 | 55 | 80 | 104 |
|---|---|---|---|---|
| decode step, ms | 47.1 | 60.0 | 69.5 | 76.7 |

A straight line fits these to within 1.6 ms: **step ≈ 36.5 ms + 0.40 ms × sessions**. Each session gets ~3 tokens a
step (one verified plus accepted drafts), so throughput ≈ 3n / (36.5 + 0.40n) tokens per ms. It reaches half its
asymptote at n ≈ 92 sessions. At 10 sessions the GPU spends 90% of each step on work that does not depend on how
many sessions are in it — reading 40 GB of expert weights, launching ~1,700 kernels. Every extra session that fits
in memory rides on that cost almost for free.

Three consequences ordered our work:
1. **Memory first.** Throughput is set by how many sessions fit, and a session at 100K context is several GB of KV.
2. **Scheduling second.** A prompt re-read from scratch costs prefill that produces no new tokens; mixed
   prefill/decode steps cost 1.3–3.4× a decode step (a 1–128-token prefill makes a step 102.8 ms against 79.9).
3. **Kernels last.** A kernel that saves 1 ms saves ~1.3% of a step.

**And throughput only counts if the model still behaves the same.** Section 9 is about how we found that out.

## 3. The path, in five phases

Every number below is a scored hidden-set submission (55 public games of the 110).

| phase | dates | what changed | board |
|---|---|---|---|
| 0. Harness work on Duck | Aug–Sep | perception, memory, prompts, delegation; 34 agent variants | 0.96–7.22 |
| 1. Speed bought with quality | Sep 8–16 | 320-expert cut by weight saliency: 25 GiB freed, KV 8→30 GiB, 2.1× tokens/s | 2.96 and 1.97 for the same kernel |
| 2. Quality first | Oct 8–11 | Franzen's agent, full 512 experts, int8 KV, 25 sessions; then the usage-ordered 400 cut | 22.75, 23.46, 21.48, 26.06 |
| 3. Lossless speed | Oct 14–30 | host KV tier, resumable sessions, graphs, drafter, kernels; 110 sessions | 25.72 → 40.63 → 45.02 → 47.19 |
| 4. Back to quality | Oct 31–Nov 1 | lossy settings reverted (int8 recurrent state, packed state, trim holes); live switch to 512 experts | 50.48 → **54.55** |

Two lessons are in this table. Phase 1 doubled throughput and moved nothing: the saliency-ordered cut drifted 3.54×
the model's noise floor and, on another model, cost 30% of the levels (61 against 87, sign test p = 0.007). Phase 2
showed the other side: on the clean serve the same agent won 41% more levels at an equal number of tokens. From then
on a speed change had to arrive without moving the model, and phases 3 and 4 are that rule applied twice — first by
finding speed that is lossless, then by spending the margin on quality.

The noise matters for reading this table. One unchanged bundle drew 22.75 and 23.46 on consecutive days; another
drew 21.48 and 26.06; early on, 2.96 and 1.97. Between Aug 18 and Sep 1 fourteen single-knob submissions read
0.82–1.93 and were all noise. A single draw cannot resolve an effect much smaller than its own spread, so we
attributed offline, on recorded sessions, and sent the strongest bundle to the board every day.

## 4. Memory: fitting 110 sessions on one GPU

### 4.1 An expert cut chosen by the router

The W4A16 model is 81.05 GiB on the GPU; at 512 experts the KV pool left for sessions is a few GiB. Cutting 112
experts per layer to 400 brings the weights to 66.42 GiB. **Which** experts to cut is the whole question:

| 400-of-512 cut ordered by | KL vs full model | × noise floor | ΔNLL |
|---|---|---|---|
| router usage on agent traffic | 0.0204 | **1.50×** | +0.0061 |
| weight saliency | 0.0481 | 3.54× | +0.0436 |

(9 recorded agent turns, 536,834 reply tokens; floor = the full model against itself at the same sampling, KL 0.0136.)

The usage order counts the router's real `topk_ids` on our own agent traffic and keeps what the agent actually
routes to. On the traffic it was ranked on, the usage-ordered 400 redirects 1.2% of routing; the saliency-ordered
400 redirects 10.3%, and a random 400 ~22%. Deeper usage cuts cost progressively more (375: 1.80× the floor, 350:
2.14×). A cut is also cheap to serve wrong: the first one missed the
MTP head's own experts, and any cut silently disabled a vendor fast path whose verification could not see it. We
cut at load time from byte ranges of the checkpoint, so one dataset serves every cut.

### 4.2 Quantized KV

Two formats, both read by our sparse-attention kernels: per-token-per-head int8 (1,040 bytes a token a layer
against 2,048 for BF16) and int4 with a randomized Hadamard rotation (~540 bytes). The pool grew from 1,983,247
tokens (int8) to 3,562,666 (int4). int4 KV is a real approximation — 2.2× the floor on its own — and we kept it
because the memory it buys is what lets 110 sessions resume instead of re-prefilling. The indexer's keys are FP8.

### 4.3 A host-RAM KV tier whose hot region is filled by the model's own attention

Each of the 12 full-attention layers is *sparse*: a lightweight indexer scores every past token and the layer
attends to the top ~2,048 KV rows. That makes a two-tier cache natural. The full KV of every session lives in
84 GiB of pinned host memory; the GPU keeps a **hot region** of 12,288 rows per session and layer. Each step the
indexer's picks are resolved against the hot region and the misses are copied over PCIe.

The design numbers came from the picks themselves (Section 11). Replayed on 13,824 recorded steps, LRU eviction
misses 6.6% of the union of picked rows at 12,288 slots; the clairvoyant Belady policy misses 4.8%, and frequency
policies do no better than LRU. Misses at step t+1 are mostly *new* picks (88% were not picked at step t), so no
cheap prefetch rule finds them: fetching what earlier layers picked this step covers 43.5% of misses at 2.7%
precision.

Three changes made the tier cheap enough for 110 sessions: one flat copy launch for all sessions that reads runs of
adjacent rows as whole 128-byte lines (0.705 → 0.565 ms a launch, byte-identical); a hash lookup instead of a sort in
the resolve step (together with the flat copy, the resolve wait fell from 5.01 to 2.49 ms at 110 sessions); and
issuing the decode-side gather before prefill attention in mixed steps, so the copy hides under prefill GEMMs.

### 4.4 Embeddings in host RAM

The model's per-layer embedding table is ~95 GiB. It lives in host RAM in a compressed store (DF11 for the
exponents, int6 in the hot cache), with a GPU cache of the hottest rows; the hot-row cache freed ~78 GiB of host RAM
that the KV tier then used.

### 4.5 A hybrid model wastes pool on padding

vLLM gives attention blocks and Gated-DeltaNet state blocks one page size. Quantizing the recurrent state to int8
left more than half of every state block as padding; packing two layers' states per block halved the state's share
of the pool (~2,090 blocks against ~4,100 for the clean BF16 state). We later turned both off on quality grounds
(Section 9).

## 5. Which experts the agent needs

The router is the model's own statement of which experts a token needs, so we asked it twice.

**Counting.** A serve patch adds the real `topk_ids` of every MoE call into one count vector per layer and writes
the tally to the log. The first count — 50,000 layer calls, ~2.1M routed tokens per layer, Duck's agent on four
games — decided the 400 cut. A second, larger trace replayed 34 recorded conversations from 25 games through the
full 512-expert model at its native top-10 routing: 2,054,643 tokens, every token labelled by kind (reasoning
1.02M, image 308K, user text 214K, system 188K, tool output 184K, tool-call code 131K), with zero misaligned
tokens against the recording.

**Routing is concentrated.** In a median layer the top 64 of 512 experts take 46.8% of the picks (uniform routing:
12.5%), the top 128 take 68.2%, the top 400 take 98.9%. The most concentrated layers are in the middle (layers 15 and
31: 69.5% and 68.7% in the top 64); the flattest is layer 0 (31.2%). A median layer has ~98 experts used at less
than a tenth of a uniform share.

**Experts specialize by kind of token, not by game.**

| comparison | top-100 expert overlap (Jaccard, mean over 48 layers) |
|---|---|
| one game against another | 0.66 (0.47–0.79 over 300 pairs) |
| half the games against the other half | 0.88 |
| image tokens against tool output | 0.48 |
| reasoning against tool-call code | 0.29 |
| reasoning against image tokens | **0.26** |

Jensen–Shannon divergence of a kind's routing against all traffic is 0.05 for reasoning, 0.10 for tool output, 0.14
for code, 0.15 for images and system text; between two games it is 0.03. Summed over layers, 7,944 of the 24,576
layer–expert pairs take at least half their mass from reasoning tokens and 3,113 from image tokens. Image tokens
route through fewer experts: at layer 47 their effective number of experts is 56 against 148 for reasoning. Prompt
and generated tokens differ too (JS 0.13, rising to 0.22 in the last layer).

**Why it matters for a cut.** A cut ranked on one kind of traffic is a cut of the experts the other kinds need. Our
400 cut was ranked on text-only traffic at contexts up to 32K. On the held-out games of the second trace it
redirects only 0.76% of the routing of *generated* tokens, but 3.25% of the routing of *prompt* tokens — 14.8% in its
worst layer — and the image tokens are where it loses most. Ranked on the mixed traffic instead, the same 400
redirects 1.77% of prompt routing and 0.74% of generated; the weight-saliency order redirects 14.3% and 7.5%. The
saliency cut that cost 30% of the levels was not unlucky: it removed experts the agent's traffic uses.

**The order of the cold experts.** The live switch (Section 10) opens all 112 cut experts at once. Its staged
variant (410, then 430, then 512 experts) opens them in the same usage order, most used first, so each partial stage
restores the experts that carry the most routing.

## 6. Scheduling for agent traffic

**Resumable sessions instead of stateless requests.** An agent turn is a long conversation plus a short append. Each
conversation keeps one paused request in the engine; the next turn extends the held sequence instead of re-reading
the prompt. This has to be exact — the engine must see the same token history the client would send — and our
first version was not: the harness dropped blank lines from the reasoning it put back into the history, so every
resumed turn diverged from the held one and re-prefilled. Replaying 1,027 recorded requests, the fix took
re-prefills of a closed session from 555 to 0. On clean runs 92–100% of turns resume.

**History trims are the dominant remaining prefill.** Franzen's agent trims its history from ~118K to ~59K tokens.
A trimmed request no longer extends the paused session, so the serve re-reads ~57K tokens. All 110 games reach the
trim within the same 20 minutes of a run, and generation falls from ~3,700 to ~2,100–2,700 tokens/s while prefill
runs at 6–8K tokens/s. We built "trim holes" — keep the recurrent state, drop the cut text from the attention KV —
which removed the re-prefill, and then reverted it on quality grounds (the recurrent state still remembers the
cut text; Section 9).

**Prefill hold with adaptive release.** Small prefills interleaved with decode create expensive mixed steps. The
scheduler holds waiting prefill while sessions decode and releases it when the token budget fills or when a cost
model fitted live says waiting costs more than merging. Mixed steps fell from 42.7% to 30.3% of all steps (measured
between two runs that also differed in trim holes, so the share is not attributed to the hold alone).

**Full CUDA graphs for mixed steps.** 3,130 of 3,136 mixed steps replayed a graph; throughput at 37 sessions went
from 430 to 611 tokens/s on the full 512-expert model.

**A processor-only multimodal cache.** The agent sends a board image every turn. Keeping the image cache only in the
API process removed a class of cache-desynchronization failures: they appeared in 0 of 212 runs with the change,
against 5 of 62 without.

**The client counts too.** The harness re-rendered the whole board history every turn; late in a game that paused
each session ~12.7 s between turns, during which its GPU seat did nothing. Rendering each board once fixed it.

## 7. Speculative decoding tuned to the workload

- **MTP drafting**, four tokens a step: per-position acceptance 0.75 / 0.56 / 0.41 / 0.31 at 110 sessions.
- **A fine-tuned draft head** (albucino's drafter, retuned on our agent traffic against the 400-expert target).
- **A restricted draft vocabulary** built from tokens the agent actually generates.
- **Verification over the top 32 logits** instead of the full 248K-wide vocabulary row: sampler time a step 2.684 →
  0.342 ms, zero mismatches on 480 rows.
- **Draft depth by measurement.** At 110 sessions and ~32K context a fourth draft adds 10% tokens a step and 10.6% to
  the decode step: a wash. A fifth raised tokens per verify step by 7.6% and lost on pure decode.

## 8. Kernels on Blackwell (SM120)

Kernel work came last on purpose, and its numbers deserve care: most were measured on an isolated bench, and on the
live serve several changed at once. The table says which.

| kernel | isolated measurement | on the live serve |
|---|---|---|
| Marlin MoE top-k summing epilogue (fp32 atomics fold the reduction into the second expert GEMM) | 0.504 → 0.439 ms a layer at 2,041 rows | mixed step MoE −2.87 ms, with other changes |
| Sparse-attention indexer, compact launch grid (row tiles of all requests back to back, not requests × longest) | — | mixed indexer 7.17 → 5.08 ms, with other changes |
| Gated-DeltaNet verify with a lookahead load | −7.6 … −21.8% a layer at 1–5 tokens, bit-exact | −0.76 ms a step at 95 sessions |
| Router, shared-expert gate and gate/up stacked into one GEMM | −0.44 ms a step at 110 rows | −1.26 … −1.31 ms with five changes |
| W4A8 on the MoE path (int8 activations, signed group scales) | 1.04× the drift floor | shipped |
| FP8 for the Gated-DeltaNet projections | 1.22× the floor | shipped (FP8 for attention projections, 1.53×, was not) |

The day's kernel work against a matched load once moved the step by about 1% (91.90 against 92.84 ms at 110
sessions, 50K context). Kernels were never where the factor of six came from.

## 9. The quality gate, and what it rejected

Every change passed a gate in two parts.

1. **Distribution drift.** KL divergence of the next-token distribution against the unmodified model on recorded
   agent turns (537K reply tokens from nine games), compared with the *noise floor* — the reference against itself
   under the same sampling. A change is described as a multiple of that floor.
2. **Behaviour.** Recorded agent sessions replayed and read: malformed tool calls, replies that end inside the
   reasoning block, re-prefills, levels at an equal token budget.

What the gate rejected or rolled back:

| change | speed | drift | why it went |
|---|---|---|---|
| batch-shared experts | +11.7% decode at 50 sessions | 2.90× | drift |
| top-9 / top-8 routing | — | 2.12× / 3.85× | drift |
| FP8 dense everywhere | — | 2.96× | drift |
| CUDA rewrite of the DeltaNet verify | 10–19% faster a layer | — | malformed tool calls 0.29% → 0.78% of replies |
| int8 recurrent state | ~2,000 pool blocks freed with packing | **1.01×** | replies ending inside the reasoning block ~3× as often |
| int6 KV + 4,096-row hot window + 2,048-token step | 110 sessions, ~10% fewer tokens/s | — | 2 levels by minute 7 against 50: malformed tool calls |
| trim holes | removed the trim re-prefill | — | the recurrent state keeps the text the trim removed |

The int8 recurrent state is the lesson. Its per-token drift was indistinguishable from the floor, and it still
broke the agent: at temperature 1.5 the replies that end inside the thinking block became about three times as
frequent, and each one costs a turn. **A per-token metric cannot see a sequence-level failure.** Only reading what
the agent did showed it.

In the last week we reverted the int8 state, the state packing and the trim holes together. Our best entry before
the revert read 47.19; the bundle with the clean serve read 50.48 (it also carried agent-side changes). The clean
serve costs almost nothing until the history trims arrive, and then pays the re-prefill of Section 6.

## 10. From speed to quality inside one run: the live expert switch

The same trade that phases 1–4 made over three months also exists inside one nine-hour run. Early, many games are
easy, each of the 110 sessions wins levels, and throughput is worth the most. Late, most sessions are stuck on hard
levels or finished; the level rate flattens and what is left needs better decisions, not more of them.

Rehearsals on the public games (110 copies: 25 games × 4 plus 10) show both halves. On the 400-expert cut with 110
sessions the run wins 210 levels in its first 30 minutes and 104 in the next 30; by minute 180–200 it wins ~22 per 30
minutes, and after minute 200 one run added 59 levels in 200 minutes (569 → 628) while its serve still generated
1,400–2,500 tokens/s. **Throughput stopped converting into levels.**

A model with all 512 experts is the better decision-maker — the 400 cut sits at 1.50× the floor — but it does not fit
beside 110 sessions, and early in the run sessions are what pays:

| play minute | 30 | 60 | 120 | 180 | 240 |
|---|---|---|---|---|---|
| 400 experts, 110 sessions (levels) | 212 | 334 | 489 | 554 | 582 (switched to 512 at 210) |
| 512 experts from minute 0, 45 sessions | 122 | 221 | 362 | 468 | 532 |
| 512 experts from minute 0, up to 60 sessions | 138 | 217 | 326 | 415 | 480 |
| tokens/s: 400 / 512@45 / 512@60 | 3,278 / 1,937 / 1,812 | 3,103 / 2,006 / 1,689 | 2,758 / 1,942 / 1,383 | 2,429 / 1,897 / 1,187 | — / 1,847 / 1,278 |

(The 400-expert tokens/s are harness-side, ~7% under the serve's own count; the others are the serve's.)

More sessions on the full model is worse, not better: at 60 sessions the KV pool sat at ~100%, 417 paused sessions
were closed under memory pressure by minute 100 (one on the 400 cut), and each turn re-read 6.3K prompt tokens against
2.8K. The full model wants fewer sessions with room to keep them; the cut model wants many sessions early.

So the serve changes model in the middle of the run, without a restart:

1. **At load** it serves the 400-expert cut, and in the background builds the 112 cold experts of every layer from
   the checkpoint into the Marlin format the kernels read (~13 GiB, written to local disk). It self-checks one
   rebuilt kept expert against the served codes byte for byte. The router is widened to 512 rows at once, with the
   cold rows held at a large negative logit, so routing is unchanged until the switch.
2. **At the switch** the harness sheds sessions down to 45 — parking the copies with the largest share of their game
   already done, and giving freed seats to those with the most left — and the scheduler drains the tail of the KV
   pool (~7,000 blocks), preempting the few requests that hold it.
3. **The worker copies the cold experts into that memory**, which is reinterpreted as expert weight views, and
   unmasks their router rows. The cold experts run as a second Marlin call through an expert map and are summed with
   the kept ones. The copy moves 12.79 GiB in 24.5–30 s; with a 15-second drain stall the whole switch completed 95 s
   after it was asked, and 45 sessions were generating 2.1–2.3K tokens/s three minutes later.

Our final public entry switched at play minute 360. Before the switch our best entry read 50.48; with it, **54.55**.

![Levels and throughput over a rehearsal run](docs/figures/run_curves.png)

*Figure 1. Cumulative levels (top) and generated tokens/s (bottom) on 110 copies of the public games. The 400-expert
run with 110 sessions is ahead from the first minutes and stays ahead; by minute ~200 its level rate has flattened
while it still generates 2,000+ tokens/s, and at minute 210 it switches live to 512 experts. The full model served
from minute 0 never catches up within 400 minutes, and it does worse with more sessions (green) than with fewer
(blue), because 60 sessions fill its KV pool.*

The schedule can be smarter than a fixed minute, and the run can measure its own plateau. Our last submission made
the switch adaptive: 410 experts at load, 430 once 15 games had finished, and 512 when the level rate plateaued —
from minute 180, checked every five minutes, switching when the levels won in the last 30 minutes fell to 8% of the
levels won in minutes 30–60, twice in a row, and at minute 360 at the latest — with the context budget raised to 150K
after the switch. On the rehearsal in Figure 1 that rule fires at minute 245 (8 levels in 30 minutes against 122 in
minutes 30–60). That submission read 41.34; it also changed five agent-side settings, and its first stage takes KV
memory at 430 experts while ~100 sessions are still playing, which is where the load would choke first.

## 11. What the model reads: attention picks as a free profiler

A sparse-attention indexer is an interpretability instrument that the model runs for you. At every decode step,
in each of the 12 attention layers, it scores every token of the context and picks the ~2,048 rows it will attend to.
The picks are already computed; logging them costs nothing. We dumped them for one long agent request — 118,173
prompt tokens, 111 messages, the 12 main attention layers, 49 consecutive decode positions, 1.2M picks — and
labelled every prompt token by where it came from.

**What the model reads, against what the prompt holds:**

| segment | share of picks | share of prompt |
|---|---|---|
| the model's own earlier reasoning | **51.9%** | 42.2% |
| tool output | **25.0%** | 17.6% |
| tool-call code it wrote | 10.5% | 10.4% |
| board images | 5.6% | 12.5% |
| user text (instructions, game frame text) | 3.8% | 12.7% |
| system prompt | 2.2% | 3.9% |

- **It reads its own notes.** Reasoning and tool output — the record of what it tried and what happened — take 77%
  of the picks. The board images get less than half their share, the instructions less than a third.
- **It reads numbers.** Tokens holding a digit are 25.2% of non-image picks against 16.6% of non-image prompt
  tokens. The recalls are scattered: coordinates, not whole boards (runs of a median 4 tokens; 0.1% of missed tokens
  sit in runs of 64 or more).
- **It reads recent text densely and old text sparsely, but it does read it.** The last two messages take 35.2% of
  picks at 1.9% of the prompt; picks 64K or more tokens back are 22.1% of the total against 45.8% of the prompt. The
  long context is used, thinly.
- **Depth changes what is read.** Layers 3–15 spend 75% of their picks on the model's reasoning; layers 19–47 spend a
  third on tool output, and 27–30% of their picks hold a digit.
- **There is little attention-sink behaviour** in this sample: the first 128 tokens get 0.8% of picks; only 56 tokens
  are picked in at least 90% of layer-steps, 52 of them reasoning.

What we did with it: the hot region's size and eviction policy (Section 4.3) were chosen on recorded picks, and the
same data killed several prefetch ideas before they cost a GPU-hour. It also explains a negative result we had no
explanation for: changes to the board image channel (outlines, renders, a perception scout) never moved the score,
and the model barely looks at the image tokens once a game is under way — what it consults is its own written
account of the board.

**Limits.** This is one request from one game at one context length, at the start of one reply. The shares above are
a measurement of that request, not a law of the model. The router trace of Section 5 — 2.05M tokens from 25 games —
points the same way from the other side: image tokens route through a small, separate set of experts, and reasoning
tokens through the widest one.

## 12. How we worked

- **Measurement before spending.** GPU time was the scarcest resource. Changes climbed a cost ladder: unit tests, a
  stub server, one paid game, three games, a 25-game rehearsal. Recorded requests were replayed offline. An archive
  of ~900 runs answered most questions before a GPU-hour was spent.
- **Attribution off the board.** With a draw spread comparable to most single improvements, the daily submission
  carried the strongest bundle and attribution happened on recorded sessions.
- **Every patch announces itself.** Each serve change is an environment knob, logs a line when it fires, and has a
  self-check that falls back to stock on a mismatch. A startup line proves installation only, so the shipped patches
  log when they fire; 336 of 492 archived serve logs carry patch lines, 102 serve-hours in total, up to 117
  concurrent sequences.
- **AI-assisted engineering.** The project ran as a team of Claude Code agents, each owning a component (the host KV
  tier, the MoE and indexer kernels, the sampler, the scheduler) under written rules; hooks blocked GPU spend nobody
  had approved, and every reported number was re-derived from its artifact before it entered the findings file.

## 13. What generalizes

1. **For agent workloads, KV residency is the product.** Session hold, prefix reuse and a host tier mattered more than
   any kernel.
2. **Let the model allocate.** The router's usage chose the experts; the indexer's picks sized the hot region; the
   level rate chose when to trade sessions for experts. Each signal is free, and each beat the hand-designed
   alternative (weight saliency, frequency eviction, a fixed model for the whole run).
3. **Measure quality where the user sees it.** A per-token divergence passed a change that broke whole replies.
4. **Build the step model first.** Two numbers — fixed cost and cost per session — told us which lever was worth
   pulling for three months.
5. **Treat evaluation noise as a first-class quantity.** If one run cannot resolve an effect, a leaderboard cannot
   either.

## 14. Code, reproduction and credits

- Code: this repository — the agent harness, the serving patches and kernels, the submission notebook.
- The serve runs on vLLM 0.30 with the patches in `serving/`; every knob of the 54.55 submission is in
  `bundle/serve/pureallk400fast_setup.sh`, and the live switch is `serving/expert_hot.py`. The 50.48 submission
  is in `submissions/vh1104-v1/`.
- Model: `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`; drafter `dfranzen/albucino-qwen3-8-flash-next-drafter`.
- Team: Florencia Solange ran experiments on her GPU quota, tuned the draft head and benchmarked the agents;
  deepdreamer and aizealotwu joined late and worked on fine-tuning.
- Built on: Daniel Franzen's milestone-2 agent and model packaging, Tufa Labs' Duck harness, Intel AutoRound,
  albucino's MTP drafter, vLLM.
