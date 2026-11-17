"""Blend two survey predictors using one held-out set of visible respondents."""

import numpy as np

import gate_expert
import statistical


GATE_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)
LOG_FLOOR = 0.001


def _choose_weight(gate_p, statistical_p, width):
    """Compare probability of the correct class under the official log floor."""
    scores = []
    for weight in GATE_WEIGHTS:
        probability = weight * gate_p + (1 - weight) * statistical_p
        scores.append(np.log(LOG_FLOOR + (1 - width * LOG_FLOOR) * probability).mean())
    best = max(scores)
    return next(w for w, score in zip(GATE_WEIGHTS, scores) if score >= best - 1e-12)


def _fit_weights(frame, schema, targets, prior):
    visible = frame.loc[frame[targets].notna().all(axis=1)].reset_index(drop=True)
    if len(visible) < 100:
        return np.ones(len(targets))
    rng = np.random.default_rng(2026)
    held = np.sort(rng.choice(len(visible), size=len(visible) // 5, replace=False))
    truth = visible.loc[held, targets].copy()
    visible.loc[held, targets] = np.nan
    gate_vectors = gate_expert.predict(visible, schema, prior)
    statistical_vectors = statistical.predict(visible, schema)
    weights = []
    for column, item in enumerate(targets):
        options = statistical._options_for(item, schema)
        codes = truth[item].map(dict(zip(options, range(len(options))))).to_numpy(dtype=int)
        # Every validation respondent had every PREDICT answer masked. Both
        # experts therefore emit exactly this same row-major target order.
        gate = np.asarray(gate_vectors[column::len(targets)])
        stat = np.asarray(statistical_vectors[column::len(targets)])
        rows = np.arange(len(held))
        weights.append(_choose_weight(gate[rows, codes], stat[rows, codes], len(options)))
    return np.array(weights)


def predict(frame, schema):
    targets = [k for k, rec in schema["items"].items() if rec["class"] == "PREDICT"]
    if not targets:
        return []
    _, blank_columns = np.nonzero(frame[targets].isna().to_numpy())
    if not len(blank_columns):
        return []
    # The schema-only prior is the same for validation and full-data inference.
    values = {name: list(schema["items"][name]["values"]) for name in targets}
    prior = gate_expert.llm_prior(schema, targets, values)
    weights = _fit_weights(frame, schema, targets, prior)
    if np.all(weights == 0):
        return statistical.predict(frame, schema)
    gate_vectors = gate_expert.predict(frame, schema, prior)
    if np.all(weights == 1):
        return gate_vectors
    stat_vectors = statistical.predict(frame, schema)
    # Both experts return normalized vectors in schema order.
    for i, w in enumerate(weights[blank_columns]):
        if w == 0.0:
            gate_vectors[i] = stat_vectors[i]
        elif w != 1.0:
            gate_vectors[i] = w * gate_vectors[i] + (1 - w) * np.asarray(stat_vectors[i])
    return gate_vectors
