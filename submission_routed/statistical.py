"""Calibrated categorical backoff submission for SimulacraBench.

The model has two deliberately simple levels:

1. A Jeffreys-smoothed item marginal is the safe fallback.
2. A categorical Naive Bayes model uses the always-visible GIVEN answers.

For every target item, a deterministic validation split selects how strongly
the contextual model may modify the marginal.  A strength of zero is an
explicit route back to the marginal, so weak context cannot make the forecast
worse merely by being available.

Training happens only on answers visible in ``frame``.  No files, network
access, hidden labels, or external data are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MARGINAL_ALPHA = 0.5
CONDITIONAL_STRENGTH = 8.0
MIN_TRAIN_ROWS = 100
MIN_VALIDATION_GAIN = 1e-3
CONTEXT_SCALES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def _options_for(name, schema):
    record = schema["items"][name]
    options = list(record["values"])
    if record.get("gate"):
        options.append(schema["gated_value"])
    return options


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _fit_model(frame, target, target_options, given_items, fit_positions):
    """Fit a smoothed categorical Naive Bayes model for one target item."""
    target_values = frame[target].to_numpy(dtype=object)[fit_positions]
    class_index = {value: index for index, value in enumerate(target_options)}
    encoded_target = np.array([class_index[value] for value in target_values], dtype=int)

    n_classes = len(target_options)
    class_counts = np.bincount(encoded_target, minlength=n_classes).astype(float)
    prior = (class_counts + MARGINAL_ALPHA)
    prior /= prior.sum()

    tables = {}
    for feature in given_items:
        values = frame[feature].to_numpy(dtype=object)[fit_positions]
        unique_values = list(pd.unique(values))
        value_index = {value: index for index, value in enumerate(unique_values)}
        encoded_values = np.array([value_index[value] for value in values], dtype=int)

        counts = np.zeros((len(unique_values), n_classes), dtype=float)
        np.add.at(counts, (encoded_values, encoded_target), 1.0)

        # Empirical-Bayes shrinkage: each class-specific feature distribution
        # is pulled toward the feature's overall distribution.  This protects
        # rare demographic cells from producing extreme probabilities.
        value_prior = np.bincount(
            encoded_values, minlength=len(unique_values)
        ).astype(float)
        value_prior /= value_prior.sum()
        probabilities = (
            counts + CONDITIONAL_STRENGTH * value_prior[:, None]
        ) / (class_counts[None, :] + CONDITIONAL_STRENGTH)

        tables[feature] = {
            value: np.log(np.maximum(probabilities[index], 1e-12))
            for value, index in value_index.items()
        }

    return prior, tables


def _evidence(frame, positions, given_items, tables, n_classes):
    """Return the class log-likelihood contributed by visible context."""
    evidence = np.zeros((len(positions), n_classes), dtype=float)
    for feature in given_items:
        values = frame[feature].to_numpy(dtype=object)[positions]
        lookup = tables[feature]
        for value in pd.unique(values):
            contribution = lookup.get(value)
            if contribution is not None:
                evidence[values == value] += contribution
    return evidence


def _choose_context_scale(frame, target, target_options, given_items, visible_positions):
    """Choose contextual strength by held-out log score on visible answers."""
    if len(visible_positions) < MIN_TRAIN_ROWS or not given_items:
        return 0.0

    # Row order is randomized by the benchmark construction.  The modulo split
    # is deterministic and avoids relying on Python's process-randomized hash.
    validation_mask = np.arange(len(visible_positions)) % 5 == 0
    validation_positions = visible_positions[validation_mask]
    fit_positions = visible_positions[~validation_mask]
    if len(validation_positions) == 0 or len(fit_positions) == 0:
        return 0.0

    prior, tables = _fit_model(
        frame, target, target_options, given_items, fit_positions
    )
    evidence = _evidence(
        frame, validation_positions, given_items, tables, len(target_options)
    )
    class_index = {value: index for index, value in enumerate(target_options)}
    truth = np.array(
        [class_index[value] for value in frame[target].to_numpy(dtype=object)[validation_positions]],
        dtype=int,
    )
    row_index = np.arange(len(validation_positions))
    log_prior = np.log(np.maximum(prior, 1e-12))[None, :]

    scores = []
    for scale in CONTEXT_SCALES:
        probabilities = _softmax(log_prior + scale * evidence)
        scores.append(float(np.log(np.maximum(probabilities[row_index, truth], 1e-12)).mean()))

    best_index = int(np.argmax(scores))
    if scores[best_index] - scores[0] < MIN_VALIDATION_GAIN:
        return 0.0
    return float(CONTEXT_SCALES[best_index])


def _predict_item(frame, target, target_options, given_items):
    values = frame[target].to_numpy(dtype=object)
    visible_positions = np.flatnonzero(~pd.isna(values))
    missing_positions = np.flatnonzero(pd.isna(values))

    scale = _choose_context_scale(
        frame, target, target_options, given_items, visible_positions
    )
    prior, tables = _fit_model(
        frame, target, target_options, given_items, visible_positions
    )
    if scale == 0.0:
        return missing_positions, np.tile(prior, (len(missing_positions), 1))

    evidence = _evidence(
        frame, missing_positions, given_items, tables, len(target_options)
    )
    probabilities = _softmax(
        np.log(np.maximum(prior, 1e-12))[None, :] + scale * evidence
    )
    return missing_positions, probabilities


def predict(frame, schema):
    """Return one probability vector per blank cell in canonical order."""
    model_items = [
        name
        for name, record in schema["items"].items()
        if record["class"] in ("GIVEN", "PREDICT")
    ]
    given_items = [
        name
        for name in model_items
        if schema["items"][name]["class"] == "GIVEN"
    ]
    predict_items = [
        name
        for name in model_items
        if schema["items"][name]["class"] == "PREDICT"
    ]

    item_predictions = {}
    for target in predict_items:
        options = _options_for(target, schema)
        missing_positions, probabilities = _predict_item(
            frame, target, options, given_items
        )
        item_predictions[target] = {
            int(position): probabilities[index].tolist()
            for index, position in enumerate(missing_positions)
        }

    vectors = []
    values = frame[model_items].to_numpy(dtype=object)
    for row in range(values.shape[0]):
        for column, item in enumerate(model_items):
            if pd.isna(values[row, column]):
                vectors.append(item_predictions[item][row])
    return vectors
