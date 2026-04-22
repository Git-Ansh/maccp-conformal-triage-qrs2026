"""
Centralized path constants for all data files and outputs.
"""

import os
from pathlib import Path

# Base project directory - src/ is under the repository root
SRC_ROOT = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = SRC_ROOT.parent  # Repository root (one level above src/)

# Data directories - data files are at repository root level
DATA_DIR = PROJECT_ROOT / "data"
ALERTS_DATA_PATH = DATA_DIR / "alerts_data.csv"
BUGS_DATA_PATH = DATA_DIR / "bugs_data.csv"
TIMESERIES_DIR = DATA_DIR / "timeseries_data" / "timeseries-data"

# Phase directories
PHASE_1_DIR = PROJECT_ROOT / "phase_1"
PHASE_2_DIR = PROJECT_ROOT / "phase_2"
PHASE_3_DIR = PROJECT_ROOT / "phase_3"
PHASE_4_DIR = PROJECT_ROOT / "phase_4"
PHASE_5_DIR = PROJECT_ROOT / "phase_5"
PHASE_6_DIR = PROJECT_ROOT / "phase_6"
PHASE_7_DIR = PROJECT_ROOT / "phase_7"

# Phase output directories
PHASE_DIRS = {
    1: PHASE_1_DIR,
    2: PHASE_2_DIR,
    3: PHASE_3_DIR,
    4: PHASE_4_DIR,
    5: PHASE_5_DIR,
    6: PHASE_6_DIR,
    7: PHASE_7_DIR
}

# Timeseries ZIP files
TIMESERIES_ZIPS = {
    'autoland1': TIMESERIES_DIR / 'autoland1.zip',
    'autoland2': TIMESERIES_DIR / 'autoland2.zip',
    'autoland3': TIMESERIES_DIR / 'autoland3.zip',
    'autoland4': TIMESERIES_DIR / 'autoland4.zip',
    'firefox-android': TIMESERIES_DIR / 'firefox-android.zip',
    'mozilla-beta': TIMESERIES_DIR / 'mozilla-beta.zip',
    'mozilla-central': TIMESERIES_DIR / 'mozilla-central.zip',
    'mozilla-release': TIMESERIES_DIR / 'mozilla-release.zip'
}

# Requirements files
REQUIREMENTS_FILES = {
    1: PROJECT_ROOT / "Phase_1_requirements.txt",
    2: PROJECT_ROOT / "Phase_2_requirements.txt",
    3: PROJECT_ROOT / "Phase_3_requirements.txt",
    4: PROJECT_ROOT / "Phase_4_requirements.txt",
    5: PROJECT_ROOT / "phase_5_requirements.txt",
    6: PROJECT_ROOT / "phase_6_requirements.txt",
    7: PROJECT_ROOT / "phase_7_requirements.txt"
}


def get_phase_output_dir(phase: int, output_type: str = 'models') -> Path:
    """
    Get output directory for a specific phase.

    Args:
        phase: Phase number (1-7)
        output_type: Type of output ('models', 'figures', 'reports', 'feature_tables')

    Returns:
        Path to the output directory
    """
    return PHASE_DIRS[phase] / 'outputs' / output_type


def ensure_dirs_exist():
    """Create all necessary directories if they don't exist."""
    for phase_dir in PHASE_DIRS.values():
        for subdir in ['src', 'notebooks', 'outputs/models', 'outputs/figures', 'outputs/reports']:
            (phase_dir / subdir).mkdir(parents=True, exist_ok=True)


# Column names for key features
ALERT_ID_COL = 'single_alert_id'
TIMESTAMP_COL = 'push_timestamp'
REGRESSION_TARGET_COL = 'single_alert_is_regression'
STATUS_TARGET_COL = 'single_alert_status'
SIGNATURE_ID_COL = 'signature_id'
REPOSITORY_COL = 'alert_summary_repository'

# Magnitude features (Phase 1)
MAGNITUDE_FEATURES = [
    'single_alert_amount_abs',
    'single_alert_amount_pct',
    'single_alert_t_value',
    'single_alert_prev_value',
    'single_alert_new_value'
]

# Context features (Phase 1)
CONTEXT_FEATURES = [
    'alert_summary_repository',
    'single_alert_series_signature_framework_id',
    'single_alert_series_signature_machine_platform',
    'single_alert_series_signature_suite',
    'single_alert_series_signature_lower_is_better',
    'alert_summary_framework'
]

# Workflow features (Phase 1)
WORKFLOW_FEATURES = [
    'single_alert_manually_created'
]

# Columns to exclude (potential leakage)
LEAKAGE_COLUMNS = [
    'single_alert_classifier',
    'single_alert_classifier_email',
    'alert_summary_first_triaged',
    'alert_summary_bug_number',
    'alert_summary_bug_updated',
    'alert_summary_bug_due_date',
    'alert_summary_notes',
    'alert_summary_status',
    'single_alert_status',  # Target in Phase 2
    'alert_summary_assignee_email',
    'alert_summary_assignee_username',
    'single_alert_starred',
    'single_alert_related_summary_id'
]

# ID columns to exclude from features
ID_COLUMNS = [
    'single_alert_id',
    'alert_summary_id',
    'signature_id',
    'alert_summary_push_id',
    'alert_summary_prev_push_id'
]

# Random seed for reproducibility
RANDOM_SEED = 42
