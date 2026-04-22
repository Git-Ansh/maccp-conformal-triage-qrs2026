#!/usr/bin/env python3
"""
Mozilla Performance Regression Detection System
Main orchestrator to run all 7 phases sequentially.

Usage:
    python main.py                  # Run all phases
    python main.py --phase 1        # Run specific phase
    python main.py --phase 1 2 3    # Run multiple phases
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent


def run_phase(phase_num: int) -> bool:
    """
    Run a specific phase.

    Args:
        phase_num: Phase number (1-7)

    Returns:
        True if successful, False otherwise
    """
    phase_dir = PROJECT_ROOT / f"phase_{phase_num}"
    run_script = phase_dir / "run.py"

    if not run_script.exists():
        print(f"[ERROR] Phase {phase_num} run script not found: {run_script}")
        return False

    print(f"\n{'='*60}")
    print(f"Running Phase {phase_num}")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(run_script)],
            cwd=str(phase_dir),
            check=True
        )
        print(f"\n[SUCCESS] Phase {phase_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Phase {phase_num} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Mozilla Performance Regression Detection System"
    )
    parser.add_argument(
        "--phase", "-p",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Phase number(s) to run. If not specified, runs all phases."
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue to next phase even if current phase fails"
    )

    args = parser.parse_args()

    # Determine which phases to run
    if args.phase:
        phases = sorted(args.phase)
    else:
        phases = [1, 2, 3, 4, 5, 6, 7]

    print(f"\n{'#'*60}")
    print("Mozilla Performance Regression Detection System")
    print(f"{'#'*60}")
    print(f"Phases to run: {phases}")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"{'#'*60}\n")

    # Run phases
    results = {}
    for phase in phases:
        success = run_phase(phase)
        results[phase] = success

        if not success and not args.skip_errors:
            print(f"\n[ABORT] Stopping due to Phase {phase} failure")
            print("Use --skip-errors to continue despite failures")
            break

    # Summary
    print(f"\n{'#'*60}")
    print("Execution Summary")
    print(f"{'#'*60}")
    for phase, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  Phase {phase}: {status}")
    print(f"{'#'*60}\n")

    # Exit code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
