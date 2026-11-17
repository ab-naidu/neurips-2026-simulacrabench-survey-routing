"""Verify the routed submission with cached Qwen weights and synthetic respondents.

From the repository root, using a CUDA-enabled environment with the declared
checkpoint already cached, run python tools/check_qwen_gpu.py data/unicef.json.
Pass additional schema paths to check all instruments in the same process.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission_routed"))
import main as candidate
import make_sandbox as sandbox


def assert_probability_vector(vector, width):
    p = np.asarray(vector)
    if (p.shape != (width,) or not np.isfinite(p).all()
            or (p < 0).any() or not np.isclose(p.sum(), 1)):
        raise ValueError("Expected a normalized finite probability vector")


def check_schema(path):
    config = sandbox.load_config(ROOT / "config.yml")
    schema = sandbox.load_schema(path, config)
    targets = sandbox.scored_items(schema)
    schema["dataset"]["n_rows"] = 121
    schema["split"] = {"n_train": 119, "n_dev": 1, "n_test": 1}
    frame = sandbox.make_sandbox(schema, config, seed=7)
    frame.loc[120, targets] = np.nan
    prior_fn = candidate.gate_expert.llm_prior
    nonuniform = []

    def checked_prior(schema, names, values):
        result = prior_fn(schema, names, values)
        if set(result) != set(names):
            raise RuntimeError("Qwen prior fallback detected or item priors missing")
        for name in names:
            assert_probability_vector(result[name], len(values[name]))
            if np.ptp(result[name]) > 1e-6:
                nonuniform.append(name)
        return result

    torch.cuda.synchronize()
    started = perf_counter()
    with patch.object(candidate.gate_expert, "llm_prior", side_effect=checked_prior) as observed:
        vectors = candidate.predict(frame, schema)
    torch.cuda.synchronize()
    if observed.call_count != 1 or not nonuniform or len(vectors) != len(targets):
        raise RuntimeError("The model prior or the routed output failed verification")
    for item, vector in zip(targets, vectors):
        assert_probability_vector(vector, len(sandbox.options_for(schema, item)))
    return dict(schema=path.stem, items=len(targets), nonuniform_priors=len(nonuniform),
                seconds_after_load=round(perf_counter() - started, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schemas", type=Path, nargs="+")
    parser.add_argument("--log", type=Path, help="Write aggregate hardware and schema measurements")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    repo = candidate.gate_expert.declared_model()
    resources = candidate.gate_expert._load_qwen(repo)
    if not resources or resources[2].device.type != "cuda":
        raise RuntimeError(f"Local CUDA load of {repo} failed; check weights and dependencies")
    model = resources[2]
    if model.config._commit_hash != candidate.gate_expert.QWEN_REVISION:
        raise RuntimeError("Loaded Qwen revision differs from candidate declaration")
    result = dict(model=repo, revision=model.config._commit_hash,
                  torch=torch.__version__, cuda_version=torch.version.cuda,
                  transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
                  dtype=str(model.dtype), load_seconds=round(perf_counter() - start, 3))
    print(json.dumps(result), flush=True)
    result["instruments"] = []
    for schema_path in args.schemas:
        measurement = check_schema(schema_path)
        result["instruments"].append(measurement)
        print(json.dumps(measurement), flush=True)
    result["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() // 2**20
    result["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() // 2**20
    print(json.dumps(result), flush=True)
    if args.log:
        args.log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
