import hashlib
import os

MOE_ROUTE_ENV = "TAAF_VLLM_MOE_ROUTE"
ROUTE_FILE_ENV = "TAAF_VLLM_MOE_ROUTE_FILE"
TRACE_REPLAY_ENV = "TAAF_VLLM_TRACE_REPLAY"
EAGER_ENV = "TAAF_VLLM_ENFORCE_EAGER"
HELPER = "moe_route.py"
INSTALLED = "vllm/taaf_moe_route.py"
ROUTE_MARKER = "vllm.taaf_moe_route"
TRACE_MARKER = "trace_decode_token_ids=(self.model_extra"
RUNNER = "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
PROTOCOL = "vllm/entrypoints/openai/completion/protocol.py"

ROUTE_FROM = (
    "            topk_weights, topk_ids = self.router.select_experts(\n"
    "                hidden_states=hidden_states,\n"
    "                router_logits=router_logits,\n"
    "                topk_indices_dtype=self._quant_method.topk_indices_dtype,\n"
    "                input_ids=input_ids,\n"
    "            )\n"
    "\n"
    "            fused_out = self.routed_experts.forward_modular(\n"
    "                x=hidden_states,\n"
    "                topk_weights=topk_weights,\n"
    "                topk_ids=topk_ids,\n"
    "                shared_experts=self._shared_experts,\n"
    "                shared_experts_input=shared_experts_input,\n"
    "            )\n")
ROUTE_TO = (
    "            topk_weights, topk_ids = self.router.select_experts(\n"
    "                hidden_states=hidden_states,\n"
    "                router_logits=router_logits,\n"
    "                topk_indices_dtype=self._quant_method.topk_indices_dtype,\n"
    "                input_ids=input_ids,\n"
    "            )\n"
    "            from vllm.taaf_moe_route import experts_range as _taaf_experts_range\n"
    "            from vllm.taaf_moe_route import route as _taaf_route\n"
    "\n"
    "            topk_weights, topk_ids = _taaf_route(\n"
    "                self.layer_name, self.router, router_logits, topk_weights, topk_ids\n"
    "            )\n"
    "\n"
    "            with _taaf_experts_range():\n"
    "                fused_out = self.routed_experts.forward_modular(\n"
    "                    x=hidden_states,\n"
    "                    topk_weights=topk_weights,\n"
    "                    topk_ids=topk_ids,\n"
    "                    shared_experts=self._shared_experts,\n"
    "                    shared_experts_input=shared_experts_input,\n"
    "                )\n")
TRACE_FROM = (
    "            thinking_token_budget=self.thinking_token_budget,\n"
    "            routed_experts_prompt_start=self.routed_experts_prompt_start,\n"
    "        )\n")
TRACE_TO = (
    "            thinking_token_budget=self.thinking_token_budget,\n"
    "            routed_experts_prompt_start=self.routed_experts_prompt_start,\n"
    "            trace_decode_token_ids=(self.model_extra or {}).get(\"trace_decode_token_ids\"),\n"
    "        )\n")
ROUTE_TARGETS = {
    RUNNER: ("05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c",
             ((ROUTE_FROM, ROUTE_TO),)),
}
TRACE_TARGETS = {
    PROTOCOL: ("5f7930635fd3924826015211032fb298206b27c7f04cd6cd2224bb128e4a3a50",
               ((TRACE_FROM, TRACE_TO),)),
}


def flag(name: str) -> bool:
    said = str(os.environ.get(name) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name} must be a boolean, found {said!r}")


def edited(site, targets: dict, marker: str, label: str) -> tuple[dict, dict]:
    made: dict = {}
    files: dict = {}
    for relative, (pinned, edits) in targets.items():
        path = site / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular vLLM source for {label}: {path}")
        stock = path.read_bytes()
        before = hashlib.sha256(stock).hexdigest()
        if marker.encode() in stock:
            files[relative] = {"patched_sha256": before}
            continue
        if before != pinned:
            raise RuntimeError(
                f"{label} cannot apply: {relative} is {before[:16]}, not the {pinned[:16]} "
                f"its edit was written against.")
        text = stock.decode("utf-8")
        for found, put in edits:
            if text.count(found) != 1:
                raise RuntimeError(
                    f"{label} cannot apply: an anchor of {relative} appears {text.count(found)} "
                    f"time(s), not once.")
            text = text.replace(found, put, 1)
        compile(text, str(path), "exec")
        made[relative] = text.encode("utf-8")
        files[relative] = {"stock_sha256": before,
                           "patched_sha256": hashlib.sha256(made[relative]).hexdigest()}
    return made, files


def landed(site, made: dict, label: str) -> None:
    for relative, data in made.items():
        path = site / relative
        temporary = path.with_name(f".{path.name}.taaf.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        if path.read_bytes() != data:
            raise RuntimeError(f"the {label} edit of {relative} did not land")


def apply(site, bundle, mtp_tokens: int) -> dict:
    route_on = flag(MOE_ROUTE_ENV)
    trace_on = flag(TRACE_REPLAY_ENV)
    row: dict = {"armed": route_on or trace_on}
    if not row["armed"]:
        return row
    route_file = str(os.environ.get(ROUTE_FILE_ENV) or "").strip()
    if route_on and not route_file:
        raise RuntimeError(
            f"{MOE_ROUTE_ENV}=1 with no {ROUTE_FILE_ENV}: the helper reads its mode from that "
            f"file, so without it every forward would stay stock while the provenance says armed.")
    if trace_on and mtp_tokens > 0:
        raise RuntimeError(
            f"{TRACE_REPLAY_ENV}=1 with {mtp_tokens} MTP token(s): vLLM refuses a replayed trace "
            f"under speculative decoding, so every replayed request would fail.")
    plans = []
    if route_on:
        helper = bundle / HELPER
        if not helper.is_file():
            raise RuntimeError(f"{MOE_ROUTE_ENV} was set but {helper} is absent")
        source = helper.read_bytes()
        compile(source.decode("utf-8"), str(helper), "exec")
        plans.append(("MOE_ROUTE", *edited(site, ROUTE_TARGETS, ROUTE_MARKER, MOE_ROUTE_ENV)))
    if trace_on:
        plans.append(("TRACE_REPLAY", *edited(site, TRACE_TARGETS, TRACE_MARKER, TRACE_REPLAY_ENV)))
    if route_on:
        installed = site / INSTALLED
        if not (installed.is_file() and installed.read_bytes() == source):
            temporary = installed.with_name(f".{installed.name}.tmp")
            temporary.write_bytes(source)
            os.replace(temporary, installed)
        row["helper_sha256"] = hashlib.sha256(source).hexdigest()
        row["route_file"] = route_file
    eager = flag(EAGER_ENV)
    row["eager"] = eager
    for label, made, files in plans:
        landed(site, made, label)
        row[label.lower()] = {"already": not made, "files": files}
    if route_on:
        graphs = ("" if eager else
                  f"; {EAGER_ENV} is off, so CUDA-graph decode steps keep the stock routing and "
                  f"only eager forwards are logged or re-picked")
        print(f"MOE_ROUTE {'already armed' if row['moe_route']['already'] else 'armed'}: every "
              f"modular MoE layer hands its top-k to {INSTALLED} {row['helper_sha256'][:16]} "
              f"after the router picks and runs its experts inside the profiler range "
              f"taaf_routed_experts; the helper reads off, log or share from {route_file} at most "
              f"once a second{graphs}", flush=True)
    if trace_on:
        print(f"TRACE_REPLAY {'already armed' if row['trace_replay']['already'] else 'armed'}: "
              f"a completion request may carry trace_decode_token_ids, and the server starts "
              f"with --enable-trace-replay, so a decode can be forced along a recorded "
              f"continuation while its real logprobs are returned", flush=True)
    return row
