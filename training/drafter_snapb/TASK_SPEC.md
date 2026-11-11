# Drafter package: tune our MTP head on much more data (2026-10-26)

## Why
Our serve decodes with speculative decoding: the model's own MTP head drafts 4 tokens, the target verifies them.
Live acceptance per position at 4 drafts is about 0.79 / 0.60 / 0.45 / 0.34, i.e. 3.17 tokens per verify step at
T 0.6. Tokens per step compound (E = 1 + c + c^2 + ... + c^n with c the conditional acceptance), so every +1% of
per-position acceptance is about +1.5% decode speed, and at higher acceptance a 5th draft starts to pay.
Speed is score for us: more tokens per game in the 9 h run.

What we tried (one RTX PRO 6000, 175 GiB RAM, Kaggle):
- tuning the dense tensors from the shipped drafter on ~1.4k calls: 2.84 -> 2.94 tokens/step at 3 drafts (+3.6%);
- more short tuning (15-20 min) on the same ~1.4k calls: flat; a second pass over 576 calls memorised;
- the MTP layer's routed experts never trained (7 updates in 15 min, the target shared the GPU);
- a bigger drafter of another family (DFlash2) was 5% worse and tuning did not close it.
We never had the memory for much more data or for the experts. You have.

## What is served now (the head to beat)
- target: Intel/Qwen3.8-Flash-Next-W4A16-AutoRound with each MoE layer cut to 400 of 512 routed experts:
  `serve/expert_usage.json` maps every layer ("0".."47" and "mtp") to its experts by usage, the first 400 are kept
  (`serve/expert_cut.py` is how our loader applies it). Hidden states for training must come from this cut model.
- drafter: the public dfranzen drafter (Kaggle model `dfranzen/albucino-qwen3-8-flash-next-drafter`, it carries the
  MTP layer's routed experts) with its 26 dense tensors replaced by `heads/mtp_hour2.safetensors` (bf16, 88.9 M params).
  `heads/mtp_k400d.safetensors` is the earlier tuned head (same layout), for reference.
- 4 drafts; draft vocabulary `serve/draft_vocab.json` (the drafter's lm_head is restricted to these ids,
  `serve/draft_vocab.py`); sampling T 0.6, top-p 0.95, top-k 20; chat template with `preserve_thinking: true`.
- The reference forward of the head (inputs: the target's last hyper-connection streams, 4 x 2560, plus the next token's
  embedding) is `MTPHead` in `carried/mtp/mtp_ref.py`, and our offline acceptance evaluator is in
  `carried/mtp/draft_ab_worker.py`, both on branch `h05-snapcurve` of this repo.

## The logs
`logs/<run>_requests.tar.gz`: our agent's recorded calls from the last three days (runs wk3, u57c, u57d, nk1, nk2:
550 game copies of the 25 public games, 33,036 calls, 2.9 GB; our earlier tuning used ~1.4k), one
`<game>-<id>_p<copy>_requests.jsonl` per game copy; counts in `logs_index.json`. Each call is two rows:
- `"event": "request"`: `messages` (system, user turns with the board picture as a base64 PNG, assistant turns with
  `reasoning` and `tool_calls`, tool results), `tools`, `chat_template_kwargs`;
- `"event": "response"`: `usage`, `finish_reason` and `spec_decode` (the live drafter's stats for that call: mean
  acceptance length, accepted / drafted tokens, histogram) — the live baseline per call.
A call's reply is the assistant message appended in the NEXT request of the same copy. Consecutive requests extend each
other until a context trim, so replaying each segment once (its last request) gives the hidden states of every reply in
it: ~10x cheaper than prefilling every call. Train only on assistant positions (reasoning + content + tool calls as
the template renders them).

Held out by game (never trained, none of their copies): dc22 lp85 s5i5 su15 wa30 (the held-out set of our earlier tuning).

## The job
1. Extract once, for all calls: the target's MTP input (4 x 2560 hyper-connection streams at every assistant
   position) and its top-20 next-token distribution at T 0.6 / top-p 0.95, from the 400-expert cut target. On 8xH100:
   two target instances at TP=4 (H200: four at TP=2), each on its own shard of the copies, writing to local NVMe (int8 or
   bf16; you have the disk, no need to drop anything).
2. Train arms from `mtp_hour2` in parallel on the free GPUs:
   - A: dense tensors only, on 1/8, 1/4, 1/2 and all train calls, the SAME number of updates for each, so the curve says
     whether more distinct data helps (one pass over all calls is a sensible update count);
   - B: dense + the MTP layer's 400 kept routed experts (from the drafter), on 1/2 and all calls — capacity vs data;
   - loss: what worked for us was total-variation toward the target's top-20 at T 0.6 plus a little soft CE;
     lr 2e-5 for the dense tensors (1e-4 was worse on every held-out game), lower for experts.
3. Evaluate every ~10% of training on the held-out games: expected tokens per step at 3 and 4 drafts with acceptance
   = sum over the draft's top-20 of min(p, q) at the target's sampling, chained per position (the same as
   `draft_ab_worker.py`); keep the best checkpoint per arm, write one JSON line per eval (arm, update, distinct calls seen,
   held-out tokens/step, per-position acceptance, the same metric on 100 of the arm's own train calls); stop an arm
   after 4 evals without +0.003.
4. Send back: the best head per arm as safetensors in the same names and shapes as `mtp_hour2` (arm B also its expert
   tensors in the drafter's naming, bf16; we quantize experts to int4 ourselves: that cost 2.9442 -> 2.9428 for us), and
   the curve file. We put any head on our serve the same day and report live tokens/s.

## How to read the curve
Flat (under +0.003 tokens/step per doubling of data for A): more data does not help the dense head. B above A: the
head was capacity-limited and the experts are the lever. Train metric far above held-out: memorising, use more data
or fewer updates.
