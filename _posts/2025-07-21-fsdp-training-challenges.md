---
layout: post
title: "FSDP Training: From Dense Models to MOE Challenges"
date: 2025-07-21
categories: [machine-learning, training, fsdp]
tags: [fsdp, moe, training, parallelism, optimization]
math: true
---

# FSDP Training: From Dense Models to MOE Challenges

Fully Sharded Data Parallel (FSDP) has become a cornerstone technique for training large language models efficiently. While FSDP works exceptionally well for dense models, the emergence of Mixture-of-Experts (MOE) architectures presents unique challenges that require careful consideration and specialized strategies.

## FSDP Success with Dense LLMs

FSDP excels at training dense language models because it can effectively scale to avoid network bandwidth bottlenecks through proper batch size selection. The key insight is that with sufficient batch size, we can fully hide communication with computation.

### Communication Cost
The communication overhead in FSDP primarily comes from:
- **All-gather operations**: Gathering sharded parameters before forward/backward passes
- **Reduce-scatter operations**: Collecting and redistributing gradients after backward passes

For modern GPU pods in datacenters, they usually have two different network channels:
- **NVLink**: High-speed interconnect for low-latency communication
- **RDMA**: Standard interconnect for higher bandwidth

Using H100 as an example, NVLink has 450 GB/s bandwidth, while RDMA has 50 GB/s bandwidth. NVLink connects 8 GPUs on one machine and RDMA connects GPUs across machines.

Both All-gather and Reduce-scatter can benefit from these two network channels:
- All-gather: To run an all-gather on more than 8 GPUs, we can first all-gather 1/8 of the data across the same GPU rank on different machines, then all-gather all data across 8 GPUs on one machine. So the communication cost can be $$ \frac{data}{8*rdma} + \frac{data}{nvlink}$$
- Reduce-scatter: To run a reduce-scatter on more than 8 GPUs, we can first reduce-scatter across 8 GPUs on one machine, then reduce-scatter 1/8 of the data across the same GPU rank on different machines. So the communication cost can be $$ \frac{data}{nvlink} + \frac{data}{8*rdma}$$

We can see that the communication costs for All-gather and Reduce-scatter are identical. In addition, since NVLink and RDMA are two different channels, they can run in parallel using pipeline optimization. As a result, the optimized communication cost is

$$
comm = max(\frac{data}{8*rdma},\frac{data}{nvlink})
$$

In modern GPU clusters, $$nvlink \approx 8*rdma$$. So we can simplify the communication cost to:

$$
comm = \frac{data}{nvlink}
$$

For a dense model with $$M$$ parameters distributed across $$N$$ devices, the communication volume per training step is:

$$\text{Communication Volume} = 2 \cdot M \cdot T_w$$

Where $$T_w$$ is the number of bytes per parameter (e.g., 2 bytes for FP16).

The communication time is:

$$T_{\text{comm}} = \frac{2 \cdot M \cdot T_w}{nvlink}$$


### Computation Cost and Batch Size

The computation time scales with batch size $$B$$:

$$T_{\text{comp}} = \frac{6 \cdot B \cdot M}{FLOPS}$$

Where 2 comes from the forward pass and 4 from the backward pass. $$B$$ is the number of tokens for each chip.

For efficient training, we need $$T_{\text{comp}} > T_{\text{comm}}$$, which gives us the minimum batch size requirement:

$$B > \frac{T_w \cdot FLOPS}{3 \cdot nvlink}$$

We can see that this is independent of the model size.

|Hardware|DType| $B$ |
|-------|-----|------------|
|H100|f32|2930|
|H100|bf16|1465|
|H100|f8|733|
|A100|f32|2080|
|A100|bf16|1040|
|H20|f32|439|
|H20|bf16|219|
|H20|f8|110|

As we can see, $$B$$ is a relatively small number that can be easily achieved.

FSDP can shard model weights, gradients, and optimizer states across all devices. So to train a larger dense model, we simply need to add more devices. 

## MOE Training Challenges with FSDP

### The MOE Scaling Problem

Mixture-of-Experts models present significant challenges for FSDP because their $$M_{comp}$$ and $$M_{params}$$ are different. FSDP communication overhead scales with $$M_{params}$$:

$$T_{\text{comm}} = \frac{2 \cdot M_{\text{params}} \cdot T_w}{nvlink}$$

But the computation scales with $$M_{comp}$$:

$$T_{\text{comp}} = \frac{6 \cdot B \cdot M_{\text{comp}}}{FLOPS}$$

This means the minimum batch size requirement becomes:

$$B > \frac{M_{\text{params}} \cdot T_w \cdot FLOPS}{3 \cdot  M_{\text{comp}} \cdot nvlink}$$

|Hardware|DType| Model | $B$ |
|-------|-----|------------| --|
|H100|f32|qwen3-32B|2930|
|H100|f32|qwen3-30B-A3B|29304|
|H100|f32|qwen3-235B-A22B|31302|
|H100|f8|deepseek-v3|16386|

`qwen3-32B` is a dense model used as a reference. As you can see, MOE models require a significantly larger $$B$$ for efficient training.

### Memory Constraints and Batch Size Limitations

MOE models consume substantial memory:
- total weights: $$\frac{M_{\text{params}} \cdot T_w}{num\_chips}$$
- total gradients: $$\frac{M_{\text{params}} \cdot T_w}{num\_chips}$$
- total optimizer state: $$\frac{2 \cdot M_{\text{params}} \cdot T_o}{num\_chips}$$ (typically $$T_o=4$$ bytes for AdamW)

With activation checkpointing, we only need the following memory for activations:

$$B \cdot D \cdot L \cdot T_{\text{act}}$$

where $$D$$ is the hidden dimension and $$L$$ is the number of layers.

Other temporary buffers for LLM training:
- 2* all-gather weights, one for current layer, one for next layer: $$\frac{2 \cdot M_{\text{params}} \cdot T_w}{L}$$
- Communication buffer + temp activations: $$15GB$$

$$B \cdot D \cdot L \cdot T_{\text{act}} + 15GB + \frac{2 \cdot M_{\text{params}} \cdot T_w}{L} + \frac{(2 \cdot T_w + 8) \cdot M_{\text{params}}}{num\_chips} < HBM$$

|Hardware|chips num|DType| Model | $B$ |
|-------|-----|------------| --|
|H100|128|f32|qwen3-235B-A22B|10145|
|H100|128|f32|qwen3-235B-A22B|10145|
|H100|256|f32|qwen3-235B-A22B|19682|
|H100|512|f32|qwen3-235B-A22B|24451|
|H100|512|bf16|qwen3-235B-A22B|64272|

It is still possible to train large MOE LLMs efficiently, but we need to tune HBM usage very carefully.

We can see that hardware with larger HBM capacity can make MOE training much simpler.

## Expert Parallelism and Gradient Accumulation

To address MOE training challenges, expert parallelism combined with gradient accumulation provides a promising solution. The key challenge of FSDP MOE is the high communication overhead for all experts. Expert parallelism + gradient accumulation allows us to run multiple mini-batch computations before performing expert communication.

### Expert Parallelism Strategy

Expert parallelism distributes different experts across different devices within a NVLink domain. For a MOE layer with $$E$$ experts across 8 GPUs:
- Each GPU hosts $$\frac{E}{8}$$ experts
- Token routing sends activations to appropriate expert GPUs
- Only active experts participate in computation for each token

Since every GPU has its own experts, we don't need to perform reduce-scatter to release expert gradients every step. We can accumulate gradients over multiple steps before reduce-scatter with other DP ranks.

This approach changes the communication overhead to:

$$T_{\text{comm}} = \frac{2 \cdot M_{\text{params}} \cdot T_w}{nvlink} + \frac{2 \cdot B \cdot topK \cdot D \cdot L \cdot T_w }{nvlink} \cdot acc\_step $$

The first part is gradient reduce-scatter and weight all-gather for each batch. The second part is token routing for MOE in each accumulation step and every layer.

And the computation cost becomes:

$$T_{\text{comp}} = \frac{6 \cdot B \cdot M_{\text{comp}}}{FLOPS} \cdot acc\_step$$

The key difference is that it multiplies by the number of mini-batches.

In this way, even given a small $$B=3000$$, we can achieve effective training since we can have a large $$acc\_step$$.


### NVLink Domain Limitations

The key limitation of this approach is HBM capacity within a NVLink domain (typically 8 GPUs). For very large MOE models:
- Total expert parameters must fit within 8×80GB = 640GB
- This constrains the maximum model size achievable with this strategy

For example, for qwen3, only $$qwen3-30B-A2B$$ ($$30*16=480GB$$) can fit in this domain.

TPU and future hardware like NVL72 can address this problem.

## Pipeline Parallelism for MOE Models

Pipeline parallelism offers another approach by splitting the model across pipeline stages. Similar to EP + Gradient accumulation, pipeline parallelism allows us to only perform expensive gradient communication after running multiple small mini-batches. It also mitigates the HBM capacity limitation for NVLink domains.

### Pipeline Parallelism Benefits

For MOE models, pipeline parallelism can:
- Distribute both shared layers and expert layers across pipeline stages
- Reduce memory requirements per device significantly
- Enable training of larger models than memory-constrained alternatives

For pipeline parallelism, we usually set $$acc\_step=PP$$. We can also shard experts for one pipeline stage within a NVLink domain to leverage high-speed NVLink bandwidth. As a result, the communication overhead for each stage of PP training becomes:

$$T_{\text{comm}} = \frac{2 \cdot M_{\text{params}} \cdot T_w}{nvlink} + \frac{2 \cdot B \cdot topK \cdot D \cdot L \cdot T_w }{nvlink} \cdot PP + \frac{2 \cdot B  \cdot D \cdot L \cdot T_w }{rdma} \cdot PP $$

This is very similar to EP + Gradient accumulation. The only difference is the last part, which is PP stage communication.

And the computation cost becomes:

$$T_{\text{comp}} = \frac{6 \cdot B \cdot M_{\text{comp}}}{FLOPS} \cdot PP$$

This is also very similar to EP + Gradient accumulation.

The memory requirement per pipeline stage becomes:

$$\frac{16 \cdot M_{\text{params}}}{PP} < 80GB \times 8$$

which is much easier to achieve than EP + Gradient accumulation.

### Pipeline Parallelism Challenges

However, pipeline parallelism introduces significant complexity:

1. **Bubble Time**: (PP-1)(FW+BW) pipeline bubble can negatively impact training efficiency. It can be reduced using technologies like interleaved pipeline or DualPipe. Usually its overhead is $$<10\%$$.
2. **Tuning Complexity**: Both FSDP and EP + Gradient accumulation are SPMD-style programming. For SPMD, every participant runs the same program, which makes it much easier to tune. On the other hand, in PP each stage runs a different program (MPMD), which can easily cause deadlocks and performance issues.
3. **Model Architecture Limitation**: Model architectures like skip connections are very challenging for PP. 

## Conclusion

FSDP training presents different challenges across model architectures:

- **Dense models**: FSDP scales naturally with achievable batch sizes.
- **MOE models**: Require careful memory tuning to preserve enough HBM for a larger batch size. Future hardware with larger HBM can make it more accessible.
- **Expert parallelism + Gradient accumulation**: Effective within NVLink domains but limited by HBM capacity. Future hardware with larger NVLink domains can make it more accessible.
- **Pipeline parallelism**: Enables larger models but significantly increases complexity and adds more restrictions.

In most cases, FSDP with some small tuning can achieve efficient MOE training.