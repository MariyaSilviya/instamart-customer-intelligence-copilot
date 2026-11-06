"""Run all project stages in sequence for the Instamart Copilot project.

Usage:
    python run_project.py

Run from any directory after installing requirements.
Raw synthetic data is generated automatically when missing.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(script_name):
    """Run a Python script and stop if it fails."""
    script_path = ROOT / script_name

    print(f"\n{'=' * 20} Running {script_name} {'=' * 20}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        text=True
    )

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    print("\nINSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    print("Starting complete project pipeline...")

    # Generate raw data if it is missing or empty.
    raw_dir = ROOT / "data" / "raw"

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print("\nRaw data not found. Generating synthetic data...")
        run_script("generate_data.py")
    else:
        print("\nRaw data found. Skipping generation.")

    # Part A: Data cleaning and SQLite storage.
    run_script("part_a_pipeline.py")

    # Part A: SQL and validation checks.
    run_script("test_queries.py")
    run_script("validate_part_a.py")

    # Part B: Customer segmentation and lapse prediction.
    run_script("part_b_pipeline.py")

    # Part C: Support ticket triage and privacy guardrails.
    run_script("part_c_pipeline.py")

    # Part D: Policy RAG and refund agent.
    run_script("part_d_pipeline.py")

    print("\n" + "=" * 60)
    print("ALL PROJECT STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nCheck these project folders:")
    print("- data/       : raw data, clean data, database and policies")
    print("- outputs/    : evaluation results and generated deliverables")
    print("- state/      : agent checkpoints")
    print("\n")


if __name__ == "__main__":
    main()