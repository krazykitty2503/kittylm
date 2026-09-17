"""Decoder-only Transformer baseline for KittyLM (plan rev 3.3, milestone B).

Modules: ``config`` (``kind: model``), ``normalization`` (RMSNorm), ``positional`` (RoPE),
``attention`` (reference + SDPA paths), ``mlp`` (SwiGLU), ``block`` (mixer seam),
``kv_cache``, ``transformer`` (``KittyLM``) and ``accounting`` (parameter budget).
This package never imports ``kittylm.training`` or ``kittylm.data`` (enforced by tests).
"""
