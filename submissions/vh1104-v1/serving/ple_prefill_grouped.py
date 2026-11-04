from __future__ import annotations

import torch
import torch.nn.functional as F

if __import__("os").environ.get("TAAF_VLLM_MARKS", "").strip() == "1":
    _marked = __import__("importlib").import_module("vllm.taaf_marks").marked
else:
    def _marked(site):
        return lambda fn: fn

NULL_BLOCK_ID = -1
GROUP_CAP_SLOTS = 65536


@_marked("ple_grouped_a")
def grouped_short_conv_prefill(
    x_p: torch.Tensor,
    q_starts: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
    num_prefills: int,
    num_prefill_tokens: int,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    has_initial_states_p: torch.Tensor,
    conv_state_len: int,
    dilation: int,
    cap: int = GROUP_CAP_SLOTS,
) -> torch.Tensor:
    output = torch.empty_like(x_p)
    if num_prefills == 0 or x_p.numel() == 0 or max_len <= 0:
        return output

    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    state_indices = state_indices_tensor_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    has_initial = has_initial_states_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.bool
    )

    per = max(1, cap // max(max_len, 1))
    weights = conv_weights.unsqueeze(1).contiguous()
    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    writing = conv_state_len > 0 and conv_state.shape[0] > 0
    gathered = (
        conv_state.new_empty((num_prefills, x_p.shape[1], conv_state_len))
        if writing else None
    )

    for start in range(0, num_prefills, per):
        stop = min(start + per, num_prefills)
        rows = stop - start
        taken = (req_indices >= start) & (req_indices < stop)
        local_rows = req_indices[taken] - start
        local_cols = col_indices[taken]

        packed = x_p.new_zeros((rows, max_len, hidden_size))
        packed[local_rows, local_cols] = x_p[taken]
        packed = packed.transpose(1, 2).contiguous()

        group_state = state_indices[start:stop]
        group_valid = valid_state[start:stop]
        group_lengths = lengths[start:stop]

        if conv_state_len > 0:
            if conv_state.shape[0] == 0:
                state = conv_state.new_zeros(
                    (rows, hidden_size, conv_state_len), dtype=x_p.dtype
                )
            else:
                state = conv_state.index_select(0, group_state)[
                    ..., :conv_state_len
                ].to(x_p.dtype)
            use_initial_mask = (
                group_valid & has_initial[start:stop]
            ).view(rows, 1, 1)
            initial_state = torch.where(
                use_initial_mask, state, torch.zeros_like(state)
            )
            history = torch.cat((initial_state, packed), dim=-1)
        else:
            history = packed

        conv_output = F.conv1d(
            history, weights, groups=history.size(1), dilation=dilation
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        valid_tokens = token_positions.view(1, max_len) < group_lengths.view(rows, 1)
        valid_output_mask = valid_tokens & group_valid.to(device=x_p.device).view(
            rows, 1
        )
        conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
        output[taken] = conv_output[local_rows, local_cols]

        if writing:
            state_starts = group_lengths.to(
                device=history.device, dtype=torch.int64
            ).view(rows, 1, 1)
            state_offsets = torch.arange(
                conv_state_len, device=history.device, dtype=torch.int64
            ).view(1, 1, conv_state_len)
            gathered[start:stop] = history.gather(
                dim=2,
                index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
            ).to(conv_state.dtype)

    if writing:
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., :conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1), gathered, existing_base_state
        )
        existing_state[..., :conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)

    return output


@_marked("ple_grouped_b")
def reference_short_conv_prefill(
    x_p: torch.Tensor,
    q_starts: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
    num_prefills: int,
    num_prefill_tokens: int,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    has_initial_states_p: torch.Tensor,
    conv_state_len: int,
    dilation: int,
) -> torch.Tensor:
    output = torch.empty_like(x_p)
    if num_prefills == 0 or x_p.numel() == 0 or max_len <= 0:
        return output

    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_p
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state_indices = state_indices_tensor_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    has_initial = has_initial_states_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.bool
    )
    if conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros(
                (num_prefills, hidden_size, conv_state_len), dtype=x_p.dtype
            )
        else:
            state = conv_state.index_select(0, state_indices)[
                ..., :conv_state_len
            ].to(x_p.dtype)
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(
            use_initial_mask, state, torch.zeros_like(state)
        )
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
        num_prefills, 1
    )
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    output.copy_(conv_output[req_indices, col_indices])

    if conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
            num_prefills, 1, 1
        )
        state_offsets = torch.arange(
            conv_state_len, device=history.device, dtype=torch.int64
        ).view(1, 1, conv_state_len)
        next_state = history.gather(
            dim=2,
            index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
        )
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., :conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        existing_state[..., :conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def peak_slots(num_prefills: int, max_len: int, cap: int = GROUP_CAP_SLOTS) -> int:
    per = max(1, cap // max(max_len, 1))
    return min(per, num_prefills) * max_len
