"""Data pipeline for KittyLM.

Stage order (the security boundary, D-006):
ingestion -> license validation -> secret scan -> quality filtering -> classification
-> deduplication -> split -> tokenization -> packed dataset.

Step 1 provides only :mod:`kittylm.data.secrets`; the remaining stages arrive in step 3.
"""
