import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submission", ROOT / "submission" / "main.py"
)
SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMISSION)


def test_predict_returns_canonical_finite_vectors():
    schema = {
        "gated_value": "NA_GATED",
        "items": {
            "country": {
                "question": "Country",
                "class": "GIVEN",
                "values": ["A", "B"],
            },
            "answer": {
                "question": "Answer",
                "class": "PREDICT",
                "values": ["yes", "no"],
            },
        },
    }
    frame = pd.DataFrame(
        {
            "respondent_id": [f"R{i}" for i in range(120)],
            "country": ["A" if i % 2 == 0 else "B" for i in range(120)],
            "answer": [
                ("yes" if i % 2 == 0 else "no") if i < 100 else np.nan
                for i in range(120)
            ],
        }
    )

    vectors = SUBMISSION.predict(frame, schema)

    assert len(vectors) == 20
    assert all(len(vector) == 2 for vector in vectors)
    assert all(np.isfinite(vector).all() for vector in vectors)
    assert all((np.asarray(vector) >= 0).all() for vector in vectors)
    assert all(np.isclose(sum(vector), 1.0) for vector in vectors)
    assert vectors[0][0] > vectors[0][1]
    assert vectors[1][1] > vectors[1][0]
