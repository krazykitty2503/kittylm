---
name: Experiment proposal
about: Propose an architecture, data or training experiment
---

## Hypothesis

<!-- What do you expect to change, and why? -->

## Change under test (one component)

## Control

<!-- Normally EXP-001 (baseline). Name the exact record. -->

## Ablation plan

- Held constant (non-embedding params, training bytes, context, schedule, ...):
- Varied:
- Metrics (bpb overall/per category, val loss, tok/s, peak VRAM, duration, ...):

## Parameter budget

<!-- Expected parameter accounting vs control. A variant must not "win" by having more
     parameters unless that is the question. -->

## Scope classification

<!-- BLOCKING / IMPORTANT / NICE_TO_HAVE / FUTURE / EXPERIMENTAL -->

## Plan authority

<!-- Which plan step does this belong to? If none, a plan revision is required first. -->
