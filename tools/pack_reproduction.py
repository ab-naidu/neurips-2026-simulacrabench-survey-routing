"""Create the local source release that accompanies the competition report."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "output" / "simulacrabench-reproduction.zip"
    output.parent.mkdir(exist_ok=True)
    # Explicitly include public schemas and score logs from synthetic runs.
    # Raw generated responses, model caches and environment files stay outside.
    paths = [ROOT / name for name in (
        "LICENSE", ".gitignore", "FAQ.md", "config.yml", "requirements.txt",
        "make_sandbox.py", "score.py", "EXPERIMENTS.md",
        "tools/check_submission_zip.py", "tools/pack_candidate.py",
        "tools/pack_reproduction.py", "tools/check_qwen_gpu.py", "tools/strict_gpu_score/main.py",
        "tests/test_submission.py", "tests/test_marginal_scores.py",
        "tests/test_gate_stress.py", "tests/test_routed.py",
        "report/reproduce.md", "report/report.tex", "report/local-results-table.tex",
        "report/official-template.tex", "report/neurips_2026.sty")]
    paths += sorted((ROOT / "data").glob("*.json"))
    paths += sorted((ROOT / "report/local-results").glob("*.json"))
    for folder in ("baseline/marginal_counts", "baseline/external_model", "submission",
                   "submission_hybrid", "submission_calibrated_hybrid", "submission_routed"):
        paths += sorted(p for p in (ROOT / folder).iterdir() if p.is_file()
                        and (p.suffix in (".py", ".txt") or p.name == "LICENSE"))
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(ROOT / "report/reproduce.md", "README.md")
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(output)


if __name__ == "__main__":
    main()
