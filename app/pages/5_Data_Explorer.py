"""Data explorer: show training and testing tables."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.components.metrics import empty_state
from app.utils.config import (
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from app.utils.data_loader import list_data_files, load_processed_data, load_raw_data, resolve_data_file

st.set_page_config(page_title=f"{PAGE_TITLE} — Data Explorer", page_icon=PAGE_ICON, layout=LAYOUT)

st.title("Data Explorer")
st.caption("Training and testing datasets")

# ── Locate files ─────────────────────────────────────────────────────────────

raw_files = list_data_files(RAW_DATA_DIR)
processed_files = list_data_files(PROCESSED_DATA_DIR)
all_files = sorted(set(raw_files + processed_files))

if not all_files:
    empty_state(
        "No data files found",
        "Place training/testing CSV files in `data/`, `data/raw/`, or `data/processed/`.",
    )
    st.stop()

def _resolve_any(filename: str):
    proc_path = resolve_data_file(PROCESSED_DATA_DIR, filename)
    if proc_path.exists():
        return proc_path, "processed"
    raw_path = resolve_data_file(RAW_DATA_DIR, filename)
    if raw_path.exists():
        return raw_path, "raw"
    return None, None


training_file = next((f for f in all_files if "training" in f.lower()), None)
testing_file = next((f for f in all_files if "testing" in f.lower()), None)

if training_file is None and len(all_files) >= 1:
    training_file = all_files[0]
if testing_file is None and len(all_files) >= 2:
    testing_file = all_files[1]

if training_file:
    training_path, training_kind = _resolve_any(training_file)
    if training_path is not None:
        st.subheader(f"Training Data — {training_file}")
        train_df = load_processed_data(training_path) if training_kind == "processed" else load_raw_data(training_path)
        st.dataframe(train_df, width="stretch", height=400)
    else:
        st.warning(f"Could not resolve training file: {training_file}")
else:
    st.info("Training CSV not found.")

if testing_file:
    testing_path, testing_kind = _resolve_any(testing_file)
    if testing_path is not None:
        st.subheader(f"Testing Data — {testing_file}")
        test_df = load_processed_data(testing_path) if testing_kind == "processed" else load_raw_data(testing_path)
        st.dataframe(test_df, width="stretch", height=400)
    else:
        st.warning(f"Could not resolve testing file: {testing_file}")
else:
    st.info("Testing CSV not found.")
