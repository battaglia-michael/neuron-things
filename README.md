# neuron-things

Model write-ups for AWS Trainium: how to run a model with the Neuron SDK, what it took, and how it performs.

| Folder | What |
|---|---|
| [nemotron-3.5-lightning-vllm-neuron](nemotron-3.5-lightning-vllm-neuron/) | NVIDIA Nemotron 3.5 Lightning 30B-A3B (hybrid Mamba-2 / MoE) on Trainium2 with vLLM + vllm-neuron and six custom NKI kernels, compared with an H100 |

Format follows [jimburtoft/NeuronStuff](https://github.com/jimburtoft/NeuronStuff). Code is Apache-2.0 (`LICENSE`);
model weights are not included.
