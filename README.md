# neuron-things

Model ports to AWS Trainium that I've led, built with AI coding agents: how to run each model with the Neuron SDK, how it
was made faster, and how it compares with GPUs, with the scripts behind every number.

| Folder | What | Result |
|---|---|---|
| [nemotron-3.5-lightning-vllm-neuron](nemotron-3.5-lightning-vllm-neuron/) | NVIDIA Nemotron 3.5 Lightning 30B-A3B (hybrid Mamba-2 / MoE) on Trainium2 with vLLM + vllm-neuron, six custom NKI kernels, compared with an H100 | Same accuracy as the H100; 5x to 11x faster than the first port; 84 % to 98 % of the H100's tokens per dollar |

Format follows [jimburtoft/NeuronStuff](https://github.com/jimburtoft/NeuronStuff). Code is Apache-2.0 (`LICENSE`);
model weights are not included.
