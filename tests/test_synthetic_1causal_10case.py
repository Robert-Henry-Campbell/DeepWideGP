import subprocess
import csv
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(
    REPO_ROOT,
    "DeepGP/results/DeepGP_snps_covs/1causal_10case/synthetic_1causal_10case_test",
)


def get_latest_results_csv(results_dir):
    csvs = glob.glob(os.path.join(results_dir, "results_*.csv"))
    assert csvs, f"No results CSV found in {results_dir}"
    return max(csvs, key=os.path.getmtime)


def parse_results(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return next(reader)


def test_synthetic_1causal_10case():
    script = os.path.join(REPO_ROOT, "scripts/test_synthetic_1causal_10case.sh")

    result = subprocess.run(
        ["bash", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    results = parse_results(get_latest_results_csv(RESULTS_DIR))

    accuracy = float(results["accuracy"])
    auc = float(results["auc"])
    specificity = float(results["specificity"])
    precision = float(results["precision"])
    recall = float(results["recall"])

    # Stable metrics — upper and lower bounds
    assert 0.85 <= accuracy <= 0.93, f"accuracy {accuracy} out of range [0.85, 0.93]"
    assert 0.72 <= auc <= 0.82, f"auc {auc} out of range [0.72, 0.82]"
    assert 0.98 <= specificity <= 1.0, f"specificity {specificity} out of range [0.98, 1.0]"

    # Sanity checks
    assert recall > 0.0, f"recall is {recall}, expected > 0"
    assert precision > 0.1, f"precision is {precision}, expected > 0.1"
    assert auc > accuracy * 0.5, f"auc {auc} should be > accuracy*0.5 ({accuracy * 0.5})"