"""Survey completion with empirical calibration of skip probabilities.

Adapted from the MIT-licensed SituatedEvals external_model example, commit
5c69e17. Its question priors and count backoff model predict substantive
answers. Visible gate outcomes estimate the probability of being asked.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models.txt")

ALPHA = 2.0        # pseudocounts carried down each backoff level
STRATA = 2         # GIVEN columns per item, most informative first
MIN_COUNT = 8      # a column with less than this to go on is not worth ranking
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GATE_STRENGTH = 8.0
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def declared_model():
    """The first Hugging Face repo id in models.txt, or None if there is none.

    The platform downloads everything named in that file into the image cache
    before the container starts, so `from_pretrained(repo_id)` resolves offline
    at runtime. Blanks and comments are skipped here, but keep the shipped file
    bare: the hosted pre-download reads it a line at a time and would try to
    fetch a comment as if it were a repo.
    """
    try:
        with open(MODELS, encoding="utf-8") as fh:
            for line in fh:
                repo = line.strip()
                if repo and not repo.startswith("#"):
                    return repo
    except OSError:
        pass
    return None


@lru_cache(maxsize=1)
def _load_qwen(repo):
    """Cache a single GPU model or its unavailable status across instruments."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo, revision=QWEN_REVISION, local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=QWEN_REVISION, torch_dtype=dtype, local_files_only=True
        ).to("cuda").eval()
        return torch, tokenizer, model
    except (ImportError, OSError, ValueError, TypeError, RuntimeError):
        return None


def llm_prior(schema, targets, values):
    """Use the cached Qwen prior when available, otherwise ordinary backoff."""
    repo = declared_model()
    resources = _load_qwen(repo) if repo else None
    if resources is None:
        return {}
    torch, tokenizer, model = resources
    device = model.device

    # Every token an option letter might arrive as, so the answer can be read
    # straight off the logits: "A" and " A" are different tokens.
    letter_ids = []
    for letter in LETTERS:
        ids = {tokenizer.encode(form, add_special_tokens=False)[0]
               for form in (letter, " " + letter)
               if tokenizer.encode(form, add_special_tokens=False)}
        letter_ids.append(sorted(ids))

    survey = schema.get("dataset", {}).get("description", "")[:600]
    prior = {}
    try:
        for item in targets:
            options = values[item][:len(LETTERS)]
            body = "\n".join(
                ["You are estimating how a survey respondent answered.", "",
                 "Survey: " + survey, "",
                 "Question: " + schema["items"][item].get("question", item), "",
                 "Options:"]
                + ["%s. %s" % (LETTERS[i], option)
                   for i, option in enumerate(options)]
                + ["", "Reply with the single letter of the most likely answer."])
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": body}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:          # not an instruct model
                prompt = body + "\nAnswer:"

            with torch.no_grad():
                logits = model(**tokenizer(prompt, return_tensors="pt").to(device)
                               ).logits[0, -1].float().cpu().numpy()

            scores = np.array([max(logits[i] for i in letter_ids[k])
                               for k in range(len(options))], float)
            p = np.exp(scores - scores.max())
            p = p / p.sum()
            if len(options) < len(values[item]):      # more than 26 options
                p = np.concatenate([p, np.full(len(values[item]) - len(options),
                                               p.mean())])
            prior[item] = p / p.sum()
    except Exception:
        return {}
    return prior


def codes(series, levels):
    """Answers as indices into `levels`; -1 for the sentinel or a blank."""
    lookup = {level: i for i, level in enumerate(levels)}
    return series.map(lookup).fillna(-1).astype(np.int64).to_numpy()


def mutual_information(x, y, nx, ny):
    """Plug-in mutual information in nats, over the rows where both are known."""
    both = (x >= 0) & (y >= 0)
    x, y = x[both], y[both]
    if x.size < MIN_COUNT:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _gate_rates(parent_obs, child_obs, levels, allowed, sentinel):
    """Estimate Pr(child was asked) for each parent level, including skips."""
    parent_codes = codes(parent_obs, levels)
    complete = (parent_codes >= 0) & child_obs.notna().to_numpy()
    asked = child_obs[complete].ne(sentinel).to_numpy(dtype=float)
    total = np.bincount(parent_codes[complete], minlength=len(levels))
    successes = np.bincount(parent_codes[complete], weights=asked, minlength=len(levels))
    group = np.isin(levels, allowed).astype(int)
    # Pool rare levels sharing the same prescribed open/closed state. Eight
    # schema pseudocounts plus half a success and a failure form the group prior.
    group_total = np.bincount(group, weights=total, minlength=2)
    group_successes = np.bincount(group, weights=successes, minlength=2)
    pooled = (group_successes + GATE_STRENGTH * np.arange(2) + 0.5) / (
        group_total + GATE_STRENGTH + 1.0
    )
    return (successes + GATE_STRENGTH * pooled[group]) / (total + GATE_STRENGTH)


def predict(frame, schema, prior):
    # ---- what the instrument is ------------------------------------------
    items = [n for n, r in schema["items"].items()
             if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if schema["items"][n]["class"] == "GIVEN"]
    targets = [n for n in items if schema["items"][n]["class"] == "PREDICT"]
    values = {n: list(schema["items"][n]["values"]) for n in items}
    gate = {n: schema["items"][n].get("gate") for n in items}

    # Gate parents before their children, so the recursion below can rely on
    # the parent's distribution already being computed.
    def depth(name):
        levels, seen = 0, set()
        while gate.get(name) and name not in seen:
            seen.add(name)
            name = gate[name]["parent"]
            levels += 1
        return levels

    # ---- the two sources of information ----------------------------------
    visible = frame.loc[~frame[targets].isna().any(axis=1)]
    # A visible question can itself have been skipped (e.g. UNICEF Q01).
    # Keep that observed category distinct from the first substantive answer.
    feature_levels = {
        g: values[g] + ([schema["gated_value"]] if gate[g] else [])
        for g in given
    }
    seen_codes = {g: codes(visible[g], feature_levels[g]) for g in given}
    card = {g: len(feature_levels[g]) for g in given}

    n = len(frame)
    frame_codes = {g: codes(frame[g], feature_levels[g]) for g in given}

    # ---- P(answer | asked), one item at a time ---------------------------
    answer = {}
    for item in targets:
        y = codes(visible[item], values[item])
        asked = y >= 0                         # the sentinel codes as -1
        width = len(values[item])

        ranked = sorted(given, reverse=True, key=lambda g: mutual_information(
            seen_codes[g][asked], y[asked], card[g], width))
        ladder = ranked[:STRATA]

        # Start from the prior, then let each backoff level pull towards the
        # counts it actually has. ALPHA is what the prior is worth in people.
        running = np.tile(prior.get(item, np.full(width, 1.0 / width)), (n, 1))
        key_fit = np.zeros(int(asked.sum()), dtype=np.int64)
        key_all = np.zeros(n, dtype=np.int64)
        size = 1
        for column in [None] + ladder:
            if column is not None:
                key_fit = key_fit * card[column] + np.maximum(
                    seen_codes[column][asked], 0)
                key_all = key_all * card[column] + np.maximum(
                    frame_codes[column], 0)
                size *= card[column]
            table = np.bincount(key_fit * width + y[asked],
                                minlength=size * width).astype(float)
            counts = table.reshape(size, width)[np.minimum(key_all, size - 1)]
            running = ((counts + ALPHA * running)
                       / (counts.sum(axis=1, keepdims=True) + ALPHA))
        answer[item] = running

    # ---- P(asked), down the gate chain -----------------------------------
    asked_prob = {}
    sentinel = schema["gated_value"]
    for item in sorted(targets, key=depth):
        if gate[item] is None:
            asked_prob[item] = np.ones(n)
            continue
        parent, allowed = gate[item]["parent"], gate[item]["observed_if"]
        child_obs = visible[item].dropna()
        marginal_asked = (child_obs.ne(sentinel).sum() + 0.5) / (len(child_obs) + 1)
        if parent not in values:
            asked_prob[item] = np.full(n, marginal_asked)
            continue
        parent_levels = values[parent] + ([sentinel] if gate[parent] else [])
        rates = _gate_rates(visible[parent], visible[item], parent_levels, allowed, sentinel)
        # When the parent is masked, marginalize over its predicted full
        # distribution. Its own skip can carry evidence about the child's skip.
        if parent in answer:
            substantive_rate = (answer[parent] * rates[:len(values[parent])]).sum(axis=1)
            probability = asked_prob[parent] * substantive_rate
            if gate[parent]:
                probability += (1.0 - asked_prob[parent]) * rates[-1]
        else:
            probability = np.full(n, marginal_asked)
        observed_parent = codes(frame[parent], parent_levels)
        known = observed_parent >= 0
        probability[known] = rates[observed_parent[known]]
        asked_prob[item] = np.clip(probability, 0.0, 1.0)

    # ---- one vector per blank cell, sentinel last ------------------------
    full = {}
    for item in targets:
        if gate[item] is None:
            full[item] = answer[item]
        else:
            p = asked_prob[item][:, None]
            full[item] = np.concatenate([p * answer[item], 1.0 - p], axis=1)

    grid = frame[items].to_numpy(dtype=object)
    return [full[items[column]][row]
            for row in range(n)
            for column in range(len(items))
            if pd.isna(grid[row, column])]
