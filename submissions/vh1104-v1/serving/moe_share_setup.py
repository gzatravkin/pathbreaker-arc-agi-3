import hashlib
import os

MOE_SHARE_ENV = "TAAF_VLLM_MOE_SHARE"
MOE_ROUTE_ENV = "TAAF_VLLM_MOE_ROUTE"
ROUTE_FILE_ENV = "TAAF_VLLM_MOE_ROUTE_FILE"
HELPER = "moe_share.py"
INSTALLED = "vllm/taaf_moe_share.py"
MARKER = "vllm.taaf_moe_share"
RUNNER = "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
STOCK_SHA256 = "05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c"

SHARE_FROM = (
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
SHARE_TO = (
    "            topk_weights, topk_ids = self.router.select_experts(\n"
    "                hidden_states=hidden_states,\n"
    "                router_logits=router_logits,\n"
    "                topk_indices_dtype=self._quant_method.topk_indices_dtype,\n"
    "                input_ids=input_ids,\n"
    "            )\n"
    "            from vllm.taaf_moe_share import route as _taaf_share_route\n"
    "\n"
    "            topk_weights, topk_ids = _taaf_share_route(\n"
    "                self.layer_name, self.router, router_logits, topk_weights, topk_ids\n"
    "            )\n"
    "\n"
    "            fused_out = self.routed_experts.forward_modular(\n"
    "                x=hidden_states,\n"
    "                topk_weights=topk_weights,\n"
    "                topk_ids=topk_ids,\n"
    "                shared_experts=self._shared_experts,\n"
    "                shared_experts_input=shared_experts_input,\n"
    "            )\n")
TARGETS = {RUNNER: (STOCK_SHA256, ((SHARE_FROM, SHARE_TO),))}


def flag(name: str) -> bool:
    said = str(os.environ.get(name) or "").strip().lower()
    if said in ("", "0", "false", "no", "off"):
        return False
    if said in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name} must be a boolean, found {said!r}")


def apply(site, bundle) -> dict:
    share_on = flag(MOE_SHARE_ENV)
    if not share_on:
        return {"armed": False}
    if flag(MOE_ROUTE_ENV):
        raise RuntimeError(
            f"{MOE_SHARE_ENV}=1 with {MOE_ROUTE_ENV}=1: both edit the same anchor of {RUNNER}, "
            f"and only one routing path may be installed in one server.")
    route_file = str(os.environ.get(ROUTE_FILE_ENV) or "").strip()
    if not route_file:
        raise RuntimeError(
            f"{MOE_SHARE_ENV}=1 with no {ROUTE_FILE_ENV}: the helper reads its k_keep, ratio and "
            f"cap from that file, so without it a load could never be re-armed without a restart.")
    helper = bundle / HELPER
    if not helper.is_file():
        raise RuntimeError(f"{MOE_SHARE_ENV} was set but {helper} is absent")
    source = helper.read_bytes()
    compile(source.decode("utf-8"), str(helper), "exec")
    path = site / RUNNER
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular vLLM source for {MOE_SHARE_ENV}: {path}")
    stock = path.read_bytes()
    before = hashlib.sha256(stock).hexdigest()
    already = MARKER.encode() in stock
    if not already:
        if before != STOCK_SHA256:
            raise RuntimeError(
                f"{MOE_SHARE_ENV} cannot apply: {RUNNER} is {before[:16]}, not the "
                f"{STOCK_SHA256[:16]} its edit was written against.")
        text = stock.decode("utf-8")
        if text.count(SHARE_FROM) != 1:
            raise RuntimeError(
                f"{MOE_SHARE_ENV} cannot apply: an anchor of {RUNNER} appears "
                f"{text.count(SHARE_FROM)} time(s), not once.")
        text = text.replace(SHARE_FROM, SHARE_TO, 1)
        compile(text, str(path), "exec")
        made = text.encode("utf-8")
        temporary = path.with_name(f".{path.name}.taaf.tmp")
        temporary.write_bytes(made)
        os.replace(temporary, path)
        if path.read_bytes() != made:
            raise RuntimeError(f"the {MOE_SHARE_ENV} edit of {RUNNER} did not land")
        patched_sha256 = hashlib.sha256(made).hexdigest()
    else:
        patched_sha256 = before
    installed = site / INSTALLED
    if not (installed.is_file() and installed.read_bytes() == source):
        temporary = installed.with_name(f".{installed.name}.tmp")
        temporary.write_bytes(source)
        os.replace(temporary, installed)
    helper_sha256 = hashlib.sha256(source).hexdigest()
    row = {"armed": True, "already": already, "route_file": route_file,
           "helper_sha256": helper_sha256,
           "moe_share": {"files": {RUNNER: {"stock_sha256": STOCK_SHA256,
                                            "patched_sha256": patched_sha256}}}}
    print(f"MOE_SHARE {'already armed' if already else 'armed'}: every main MoE layer hands its "
          f"top-k to {INSTALLED} {helper_sha256[:16]} under CUDA graphs and MTP, before "
          f"{RUNNER.rsplit('/', 1)[-1]} runs the fused expert kernel; k_keep, ratio and cap live in "
          f"a device tensor read from {route_file} at most once a second, so a load needs no "
          f"server restart", flush=True)
    return row
