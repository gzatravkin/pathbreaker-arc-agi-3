TARGETS = {
    'vllm/v1/worker/gpu/model_states/mamba_hybrid.py': (
        '573e130ac5587d4822fabf97adad1ad0fda02ff83c5e70df879a5bb175ae845c',
        (
            (
                ('import torch.nn as nn\n'),
                ('import torch.nn as nn\n'
                 'from vllm import taaf_gdn_commit_stream\n'),
            ),
            (
                ('        if self.recoverssm is not None:\n'
                 '            self.recoverssm.commit_step(\n'),
                ('        _commit_side = taaf_gdn_commit_stream.begin(\n'
                 '            self, idx_mapping, num_sampled, num_computed_tokens\n'
                 '        )\n'
                 '        if self.recoverssm is not None:\n'
                 '            self.recoverssm.commit_step(\n'),
            ),
            (
                ('        if not num_reqs:\n'
                 '            return\n'),
                ('        if not num_reqs:\n'
                 '            taaf_gdn_commit_stream.end(_commit_side)\n'
                 '            return\n'),
            ),
            (
                ('                num_computed_tokens,\n'
                 '                idx_mapping,\n'
                 '            )\n'),
                ('                num_computed_tokens,\n'
                 '                idx_mapping,\n'
                 '            )\n'
                 '        taaf_gdn_commit_stream.end(_commit_side)\n'),
            ),
        ),
    ),
    'vllm/v1/worker/gpu/model_runner.py': (
        '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
        (
            (
                ('        if not dummy_run:\n'
                 '            # Update the request states.\n'),
                ('        from vllm import taaf_gdn_commit_stream\n'
                 '\n'
                 '        taaf_gdn_commit_stream.join()\n'
                 '        if not dummy_run:\n'
                 '            # Update the request states.\n'),
            ),
            (
                ('                        with use_workspace_lane(self._draft_workspace_lane):\n'
                 '                            self.speculator.capture()\n'),
                ('                        from vllm import taaf_gdn_commit_stream\n'
                 '\n'
                 '                        with use_workspace_lane(\n'
                 '                            self._draft_workspace_lane\n'
                 '                        ), taaf_gdn_commit_stream.draft_lane(capture=True):\n'
                 '                            self.speculator.capture()\n'),
            ),
            (
                ('            with use_workspace_lane(self._draft_workspace_lane):\n'
                 '                draft_tokens = self.speculator.propose(\n'),
                ('            from vllm import taaf_gdn_commit_stream\n'
                 '\n'
                 '            with use_workspace_lane(\n'
                 '                self._draft_workspace_lane\n'
                 '            ), taaf_gdn_commit_stream.draft_lane():\n'
                 '                draft_tokens = self.speculator.propose(\n'),
            ),
        ),
    ),
}
