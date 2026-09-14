# KittyLM Scaling Roadmap

Scaling happens in stages. Sizes listed for future stages are targets to be justified by
memory and compute measurements, not measured claims.

> **Rule: never introduce distributed training until single-device training and checkpoint
> recovery are demonstrably correct.**

| Stage | Hardware | Model scale | Techniques introduced |
|---:|---|---|---|
| 0 (current) | 1× AMD Radeon RX 9060 XT, 16 GB (ROCm) | `tiny` config ≈ 17M parameters (16,913,280 by construction at vocab 16,384) | single device, bf16, gradient accumulation |
| 1 | same single GPU | targets ~32M / ~70M / ~125M, chosen by measured memory and throughput | larger configs, longer runs |
| 2 | same single GPU | as large as memory allows | activation checkpointing, tuned gradient accumulation |
| 3 | multiple GPUs | beyond one device's throughput | DDP |
| 4 | multiple GPUs / nodes | beyond one device's memory | FSDP, tensor parallelism, pipeline parallelism |
| 5 | cloud / HPC | research-scale | managed clusters |

## Entry criteria for each stage

- **Stage 1:** EXP-001 recorded; checkpoint resume test `passed` with harness evidence.
- **Stage 2:** a measured memory limit is hit at Stage 1.
- **Stage 3:** single-device training, checkpoint recovery and evaluation are demonstrably
  correct at the largest single-GPU scale, and throughput (not memory) is the bottleneck.
- **Stage 4:** a model that is justified by ablation evidence no longer fits on one device.
- **Stage 5:** local and multi-GPU options are exhausted for a question worth answering.

Nothing beyond Stage 0 is in scope for KittyLM 0.1.
