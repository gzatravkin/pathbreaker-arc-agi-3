TARGETS = {
    'vllm/v1/hisparse/runtime.py': (
        None,
        (
            ('        device_global_indices = self.index_group.device_global_indices\n        active_indices = device_global_indices.index_select(0, state_indices)\n        positions = torch.searchsorted(sorted_slots, active_indices)\n',
             '        device_global_indices = self.index_group.device_global_indices\n        from vllm import taaf_small_kernels\n\n        if taaf_small_kernels.invalidate_sorted(device_global_indices, sorted_slots, state_indices):\n            return\n        active_indices = device_global_indices.index_select(0, state_indices)\n        positions = torch.searchsorted(sorted_slots, active_indices)\n'),
        ),
    ),
}
