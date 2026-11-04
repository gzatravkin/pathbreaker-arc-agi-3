TARGETS = {
    'vllm/v1/worker/gpu/cudagraph_utils.py': (
        '6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff',
        (
            (
                ('        self.graphs[desc].replay()\n'),
                ('        from vllm import taaf_ple_hot\n'
                 '\n'
                 '        taaf_ple_hot.replay(self.graphs[desc])\n'),
            ),
            (
                ('                        graph = torch.cuda.CUDAGraph()\n'),
                ('                        from vllm import taaf_ple_hot\n'
                 '\n'
                 '                        graph = taaf_ple_hot.new_graph(self.pool, self)\n'),
            ),
            (
                ('                        with torch.cuda.graph(\n'
                 '                            graph, self.pool, stream=self._capture_stream(desc)\n'
                 '                        ):\n'),
                ('                        with taaf_ple_hot.capturing(\n'
                 '                            graph, self.pool, self._capture_stream(desc)\n'
                 '                        ):\n'),
            ),
        ),
    ),
}
