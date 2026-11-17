"""Local score.py adapter for submission_routed; fail if Qwen falls back."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "submission_routed"
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location("routed_candidate", SOURCE / "main.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
prior_fn = candidate.gate_expert.llm_prior


def require_qwen(schema, targets, values):
    priors = prior_fn(schema, targets, values)
    if set(priors) != set(targets):
        raise RuntimeError("Qwen prior missing or incomplete")
    for item in targets:
        vector = np.asarray(priors[item])
        if (vector.shape != (len(values[item]),) or not np.isfinite(vector).all()
                or (vector < 0).any() or not np.isclose(vector.sum(), 1)):
            raise ValueError("Invalid Qwen prior")
    if not any(np.ptp(p) > 1e-6 for p in priors.values()):
        raise ValueError("Expected a non-uniform Qwen prior on this public instrument")
    return priors


def predict(frame, schema):
    with patch.object(candidate.gate_expert, "llm_prior", side_effect=require_qwen) as observed:
        results = candidate.predict(frame, schema)
    if observed.call_count != 1:
        raise RuntimeError("Candidate skipped prior computation")
    return results
