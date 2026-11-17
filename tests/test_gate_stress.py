"""Paired rehearsal with invented survey answers. Run directly from the project root."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import score
from make_sandbox import load_config


SCHEMA = {
    "gated_value": "NA_GATED",
    "items": {
        "eligible": {"class": "GIVEN", "values": ["yes", "no"]},
        "interest": {"class": "PREDICT", "values": ["yes", "no"],
                     "gate": {"parent": "eligible", "observed_if": ["yes"]}},
        "frequency": {"class": "PREDICT", "values": ["daily", "weekly"],
                      "gate": {"parent": "interest", "observed_if": ["yes"]}},
    },
}
SEEDS = (41, 42, 43)


def respondents(seed, noisy):
    """One visible parent, then a hidden parent that can itself be skipped."""
    rng = np.random.default_rng(seed)
    n = 6000
    eligible = rng.random(n) < 0.58
    # In the noisy arm, some ineligible respondents are surveyed and some
    # eligible respondents are skipped. Sampling uses exactly the same variates
    # across arms. No fitted or external data enter this generator.
    interest_rate = np.where(eligible, 0.85, 0.22) if noisy else eligible
    interest_asked = rng.random(n) < interest_rate
    interest = np.where(interest_asked, np.where(rng.random(n) < 0.62, "yes", "no"),
                        SCHEMA["gated_value"])
    frequency_rate = (np.select([interest == "yes", interest == "no"], [0.81, 0.12],
                                default=0.31) if noisy else interest == "yes")
    frequency_asked = rng.random(n) < frequency_rate
    frequency = np.where(frequency_asked, np.where(rng.random(n) < 0.40, "daily", "weekly"),
                         SCHEMA["gated_value"])
    return pd.DataFrame({
        "respondent_id": [f"r{i}" for i in range(n)],
        "role": np.repeat(["TRAIN", "DEV", "TEST"], [3000, 1000, 2000]),
        "eligible": np.where(eligible, "yes", "no"),
        "interest": interest,
        "frequency": frequency,
    })


def load_candidate(directory):
    spec = importlib.util.spec_from_file_location("gate_model", ROOT / directory / "main.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def test_gate_stress():
    models = {"hard": load_candidate("baseline/external_model"),
              "calibrated": load_candidate("submission_calibrated_hybrid")}
    floor = load_config(str(ROOT / "config.yml"))["scoring"]["floor"]
    for noisy in (False, True):
        scores = {name: [] for name in models}
        for seed in SEEDS:
            frame, cells, truth = score.sample_rows(SCHEMA, respondents(seed, noisy), phase=2)
            for name, model in models.items():
                with patch.object(model, "llm_prior", return_value={}):
                    vectors = model.predict(frame.copy(), SCHEMA)
                score.check(SCHEMA, vectors, cells)
                vectors = score.floored(vectors, SCHEMA, cells, floor)
                per_item = {"interest": [], "frequency": []}
                for vector, (_, _, item), actual in zip(vectors, cells, truth):
                    index = score.options_for(SCHEMA, item).index(actual)
                    per_item[item].append(np.log(vector[index]))
                scores[name].append([np.mean(per_item[item]) for item in per_item])
        condition = "noisy" if noisy else "exact"
        for name, values in scores.items():
            direct, recursive = np.mean(values, axis=0)
            print(f"{condition:5} {name:10} overall={np.mean(values):.6f} "
                  f"visible_parent={direct:.6f} hidden_parent={recursive:.6f}")
        gains = np.asarray(scores["calibrated"]) - np.asarray(scores["hard"])
        if noisy:
            assert (gains > 0).all(), "Both gate types should improve on each noisy seed."
        else:
            assert (gains >= -0.01).all(), "Exact-rule regression exceeds 0.01 nats per cell."


def test_visible_skip_is_a_feature_level():
    # In a survey with approximate gates, a GIVEN field can carry the skip
    # sentinel even within the same parent stratum. It must not alias "first".
    items = {
        "region": {"class": "GIVEN", "values": ["one"]},
        "status": {"class": "GIVEN", "values": ["first", "second"],
                   "gate": {"parent": "region", "observed_if": ["one"]}},
        "outcome": {"class": "PREDICT", "values": ["A", "B"]},
    }
    sentinel = "NA_GATED"
    frame = pd.DataFrame({"region": ["one"] * 202,
                          "status": ["first", sentinel] * 101,
                          "outcome": ["A", "B"] * 100 + [np.nan, np.nan]})
    model = load_candidate("submission_calibrated_hybrid")
    with patch.object(model, "llm_prior", return_value={}):
        first, skipped = model.predict(frame, {"items": items, "gated_value": sentinel})
    assert first[0] > 0.9 and skipped[1] > 0.9


if __name__ == "__main__":
    test_visible_skip_is_a_feature_level()
    test_gate_stress()
