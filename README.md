# Survey Routing at the NeurIPS 2026 SimulacraBench Competition

Survey Routing predicts missing survey responses by estimating how often a follow-up question is asked and learning how much to rely on that estimate for each question. Our final submission, `1057678`, earned a displayed overall skill score of **0.35**. [Official Test leaderboard](https://www.codabench.org/api/phases/29773/get_leaderboard/?format=json)

Run the commands below from the repository root. Our MIT-licensed submission is in `submission_routed/`. The organizers supply the grader, schemas and synthetic generator. The root and submission licenses preserve the original SituatedEvals notice. The upstream repository also includes the original tutorial notebooks and optional baselines. [Pinned organizer starter kit](https://github.com/SituatedEvals/public/tree/5c69e17b191e82fe867eea6daf06c8ba8feaf622)

## Rebuild the scored archive

The first command rebuilds the ZIP with the Python standard library. The second checks its layout. The expected SHA-256 for submission `1057678` is `0bd3220a562815c8568c6943dbe9ad639274f94389c1f14fa7fc31bca997f84e`.

```text
python tools/pack_candidate.py submission_routed submission_routed.zip
python tools/check_submission_zip.py submission_routed.zip
```

## Local CPU evaluation

In a fresh Python environment, run `python -m pip install -r requirements.txt` to install the generator and grader dependencies. The recorded CPU runs used Python 3.10.11, NumPy 2.2.6, pandas 2.3.3, PyYAML 6.0.3 and PyTorch 2.14.1+cpu. The generator and grader also support CSV when pyarrow is unavailable. The CPU branch can run without PyTorch. The hosted ZIP imports libraries supplied in the image.

```powershell
foreach ($seed in 0,17) {
  foreach ($name in 'unicef','world_bank','unhcr') {
    python make_sandbox.py --schema "data/$name.json" --out "_sandbox/replay-$seed/$name" --seed $seed
    python score.py --submission submission --data "_sandbox/replay-$seed/$name" --schema "data/$name.json" --phase 2
    python score.py --submission submission_routed --data "_sandbox/replay-$seed/$name" --schema "data/$name.json" --phase 2
  }
}
python tests/test_gate_stress.py
python tests/test_routed.py
```

Use `submission_hybrid` for the organizer hard-gate reference and `submission_calibrated_hybrid` for the corresponding empirical-gate ablation. For a matched no-Qwen comparison, run in a CPU-only Python environment. The stress test explicitly disables Qwen in both arms. Its artificial gate-deviation rates are documented in the test.

## GPU verification

The checkpoint is `Qwen/Qwen2.5-1.5B-Instruct` at public revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. On a CUDA machine with that revision cached, run:

```text
python tools/check_qwen_gpu.py data/unicef.json data/world_bank.json data/unhcr.json
```

This check verifies the loaded model revision and requires valid, nonuniform priors on each public instrument. To score the full seed-0 datasets created above with the official grader and enforce actual GPU inference, run the following from the same environment.

```powershell
foreach ($name in 'unicef','world_bank','unhcr') {
  python score.py --submission tools/strict_gpu_score --data "_sandbox/replay-0/$name" --schema "data/$name.json" --phase 2
}
```

The recorded local GPU check used an RTX 5060 Laptop GPU under WSL, Python 3.10.21, NumPy 1.26.4, pandas 2.3.3, PyTorch 2.7.1+cu128, Transformers 4.53.3, and huggingface-hub 0.34.4. The checkpoint loaded in bfloat16. The small-frame check (`report/local-results/gpu-smoke.json`) assesses model loading, the prior for every public question and the shape of the routed predictions. Full synthetic grader logs are separate.

The local adapter calls the submitted predictor and raises an error if Qwen is unavailable. The competition code uses a uniform prior when GPU inference is unavailable. Codabench reads the bare repository name from `models.txt` during its build step. At inference the loader requests the fixed revision with local files only. If the hosted cache lacks that revision, the uniform fallback is used. Model files are obtainable from the [pinned Hugging Face snapshot](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).

## Report

`report/report.tex` contains our competition report. Build with `tectonic report/report.tex --outdir output/pdf` after creating the output directory, or run `pdflatex report.tex` twice in `report/`. The organizer's original template is preserved in `report/official-template.tex`. [Venue template](https://drive.google.com/file/d/1ic9SiOA9m11Qcgls5_twByPnpfnse7M5/view)
