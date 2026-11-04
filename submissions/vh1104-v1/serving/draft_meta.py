import os
import sys

DRAFT_META_ENV = "TAAF_VLLM_DRAFT_FUSED_META"
COMMON_FIELD = "_taaf_draft_common"
MARK = "_taaf_draft_meta"
_SAID = set()
_TALLY = {"builds": 0, "updates": 0, "skipped": 0}


def armed():
    return str(os.environ.get(DRAFT_META_ENV) or "").strip().lower() in ("1", "on", "true")


def say(key, line):
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, flush=True)


def tally():
    return dict(_TALLY)


def _common(args, kwargs):
    if "common_attn_metadata" in kwargs:
        return kwargs["common_attn_metadata"]
    return args[1] if len(args) > 1 else None


def rebuild(builder, metadata):
    common = getattr(metadata, COMMON_FIELD, None)
    if common is None or metadata.num_prefills or not metadata.num_decode_tokens:
        _TALLY["skipped"] += 1
        return False
    build_k_work = not builder.is_circular_buffer and builder.compress_ratio != 1
    sys.modules[type(builder).__module__].build_qsa_metadata(
        common,
        builder.token_to_req_buffer,
        builder.logical_positions_buffer,
        builder.visible_blocks_buffer,
        builder.slot_mapping_buffer,
        storage_block_size=builder.storage_block_size,
        compress_ratio=builder.compress_ratio,
        circular_buffer_size=builder.kv_cache_spec.block_size if builder.is_circular_buffer else 0,
        k_work_metadata_buffer=metadata.k_work_metadata if build_k_work else None,
        request_capacity=builder.request_capacity if build_k_work else None,
    )
    _TALLY["updates"] += 1
    say("live", "DRAFT_FUSED_META live: a draft decode step rebuilt its QSA side-cache metadata in place from "
                "device tensors, so draft steps 1+ run as one fused loop instead of a host rebuild between steps")
    return True


def install(builder_class):
    if not armed() or getattr(builder_class, MARK, False):
        return False
    build = builder_class.build

    def kept_build(self, *args, **kwargs):
        metadata = build(self, *args, **kwargs)
        object.__setattr__(metadata, COMMON_FIELD, _common(args, kwargs))
        _TALLY["builds"] += 1
        return metadata

    def update_draft_decode_metadata(self, metadata):
        rebuild(self, metadata)

    builder_class.build = kept_build
    builder_class.update_draft_decode_metadata = update_draft_decode_metadata
    builder_class.supports_draft_decode_metadata_update = True
    setattr(builder_class, MARK, True)
    say("armed", f"DRAFT_FUSED_META armed: {builder_class.__name__} declares in-place draft metadata updates, "
                 f"so the drafter may take the fused multi-step decode path")
    return True
