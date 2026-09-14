## Summary

<!-- What changed and why. Which plan step does this implement? -->

## Checklist

- [ ] `python scripts/check.py all` passes locally
- [ ] Required CI jobs pass
- [ ] GPU tests (`pytest -m gpu`) run locally: yes / no / not applicable
- [ ] Any reported result has a validated `experiments/<ID>/record.yaml` (no hand-typed numbers)
- [ ] `experiments/ablations.md` regenerated if records changed
- [ ] Any new data source has license + provenance recorded
- [ ] No secrets, `.env` files, data, checkpoints, weights, samples or logs are committed
- [ ] Module docstrings follow the documentation rule (Purpose, Public API, Invariants,
      Failure modes, See; plus Shapes/Dtype/Device for tensor modules)
- [ ] Within the approved plan (otherwise: plan revision first)

## What was tested / measured

## What remains uncertain
