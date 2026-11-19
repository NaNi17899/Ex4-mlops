"""
Module: dvc_versioning_demo.py
Description: Demonstrates dataset artifact tracking, DVC remote push/pull,
             and version switching between v1 (5,000 rows) and v2 (7,043 rows) using Git & DVC.
"""

import os
import sys
import subprocess
import pandas as pd

PYTHON_EXE = f'"{sys.executable}"'

def run_cmd(cmd: str):
    print(f">> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr and res.returncode != 0:
        print(f"Stderr: {res.stderr.strip()}")
    return res

def setup_dvc_versions():
    print("=" * 60)
    print(">>> DVC & GIT DATASET ARTIFACT VERSIONING SETUP <<<")
    print("=" * 60)

    raw_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    dataset_path = "data/dataset.csv"
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}")
        
    df_full = pd.read_csv(raw_path)
    
    # -------------------------------------------------------------
    # Version 1: 5000 records baseline
    # -------------------------------------------------------------
    print("\n[Step 1] Creating Dataset Version 1 (5,000 samples)...")
    df_v1 = df_full.iloc[:5000]
    df_v1.to_csv(dataset_path, index=False)
    print(f"Dataset v1 written: {len(df_v1)} rows.")
    
    run_cmd(f'{PYTHON_EXE} -m dvc add data/dataset.csv')
    run_cmd(f'{PYTHON_EXE} -m dvc push')
    run_cmd('git add data/dataset.csv.dvc .dvc/config .gitignore')
    run_cmd('git commit -m "DVC Track dataset v1 (5000 rows)" --allow-empty')
    run_cmd('git tag -f -a v1.0 -m "Dataset v1 baseline"')
    print("✓ Dataset v1 successfully tracked and committed to Git tag v1.0")

    # -------------------------------------------------------------
    # Version 2: Full 7043 records production
    # -------------------------------------------------------------
    print("\n[Step 2] Updating Dataset Version 2 (Full 7,043 samples)...")
    df_full.to_csv(dataset_path, index=False)
    print(f"Dataset v2 written: {len(df_full)} rows.")
    
    run_cmd(f'{PYTHON_EXE} -m dvc add data/dataset.csv')
    run_cmd(f'{PYTHON_EXE} -m dvc push')
    run_cmd('git add data/dataset.csv.dvc')
    run_cmd('git commit -m "DVC Track dataset v2 (7043 rows)" --allow-empty')
    run_cmd('git tag -f -a v2.0 -m "Dataset v2 production"')
    print("✓ Dataset v2 successfully tracked and committed to Git tag v2.0")

def demonstrate_version_switching():
    print("\n" + "=" * 60)
    print(">>> DEMONSTRATING REPRODUCIBILITY & VERSION SWITCHING <<<")
    print("=" * 60)

    dataset_path = "data/dataset.csv"
    
    # Check current (v2)
    cur_len = len(pd.read_csv(dataset_path))
    print(f"\nCurrent Active Dataset Rows (v2): {cur_len}")
    
    # Switch to v1.0
    print("\nSwitching to Version 1 (git checkout v1.0 && dvc checkout)...")
    run_cmd('git checkout v1.0 -- data/dataset.csv.dvc')
    run_cmd(f'{PYTHON_EXE} -m dvc checkout data/dataset.csv.dvc')
    v1_len = len(pd.read_csv(dataset_path))
    print(f"Verified Active Dataset Rows after checkout: {v1_len} (Expected: 5000)")
    assert v1_len == 5000, f"Expected 5000, got {v1_len}"
    
    # Switch back to v2.0
    print("\nSwitching back to Version 2 (git checkout v2.0 && dvc checkout)...")
    run_cmd('git checkout v2.0 -- data/dataset.csv.dvc')
    run_cmd(f'{PYTHON_EXE} -m dvc checkout data/dataset.csv.dvc')
    v2_len = len(pd.read_csv(dataset_path))
    print(f"Verified Active Dataset Rows after checkout: {v2_len} (Expected: 7043)")
    assert v2_len == 7043, f"Expected 7043, got {v2_len}"

    print("\n✓ Version switching & reproducibility demonstration completed successfully!")

if __name__ == "__main__":
    setup_dvc_versions()
    demonstrate_version_switching()
