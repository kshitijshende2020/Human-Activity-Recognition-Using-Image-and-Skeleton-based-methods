import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict

def main():
    # Base paths relative to project root (HAR-Image Based/)
    project_root = Path(__file__).resolve().parent.parent
    manifests_dir = project_root / "manifests"
    split_csv_path = manifests_dir / "split_manifest.csv"  # Reusing original subject split manifest
    log_csv_path = manifests_dir / "extraction_log_3frame.csv"
    data_split_3frame_dir = project_root / "data_split_3frame"

    # Input validation
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_csv_path}")
    if not log_csv_path.exists():
        raise FileNotFoundError(f"Missing 3-frame extraction log: {log_csv_path}")

    # 1. Read existing subject_id -> split mapping from split_manifest.csv
    subject_to_split = {}
    with open(split_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_to_split[row["subject_id"].strip()] = row["split"].strip()

    # 2. Read extraction_log_3frame.csv
    extraction_rows = []
    actions_set = set()
    with open(log_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            extraction_rows.append(row)
            actions_set.add(row["action"].strip())

    actions_list = sorted(list(actions_set))

    # 3. Clean data_split_3frame/ safely if it exists from a previous run
    if data_split_3frame_dir.exists():
        print(f"Cleaning previous contents in: {data_split_3frame_dir}")
        shutil.rmtree(data_split_3frame_dir)

    # Re-create target split/action subfolders
    for split in ["train", "val", "test"]:
        for action in actions_list:
            (data_split_3frame_dir / split / action).mkdir(parents=True, exist_ok=True)

    # Metrics tracking
    split_counts = defaultdict(int)
    matrix_counts = defaultdict(lambda: defaultdict(int))  # action -> split -> count
    copied_count = 0
    missing_matches = 0
    missing_source_files = 0

    print("=" * 65)
    print("Stage 3 (3-Frame Pipeline) — Dataset Materialization Starting")
    print(f"Total 3-frame records to materialize : {len(extraction_rows)}")
    print(f"Target Output Directory                : {data_split_3frame_dir}")
    print("=" * 65)

    # 4. Copy images from frames_raw_3frame/ to data_split_3frame/<split>/<action>/
    for row in extraction_rows:
        subj_id = row["subject_id"].strip()
        action_name = row["action"].strip()
        rel_img_path = row["output_image"].strip()  # e.g. frames_raw_3frame/backhand/p10_backhand_s1_frame1.jpg

        source_img_path = project_root / rel_img_path

        if not source_img_path.exists():
            print(f"[ERROR] Source file missing: {source_img_path}")
            missing_source_files += 1
            continue

        split_name = subject_to_split.get(subj_id)
        if not split_name:
            print(f"[ERROR] Subject '{subj_id}' not found in split_manifest.csv!")
            missing_matches += 1
            continue

        dest_img_path = data_split_3frame_dir / split_name / action_name / source_img_path.name

        shutil.copy2(source_img_path, dest_img_path)

        copied_count += 1
        split_counts[split_name] += 1
        matrix_counts[action_name][split_name] += 1

    # 5. Output Summary and Breakdown Table
    print("\n" + "=" * 65)
    print("STAGE 3 (3-FRAME PIPELINE) MATERIALIZATION COMPLETE")
    print("=" * 65)
    print(f"Total Records Target    : {len(extraction_rows)}")
    print(f"Total Images Copied     : {copied_count} (Expected ~5,940)")
    print(f" - Train Images        : {split_counts['train']}")
    print(f" - Val Images          : {split_counts['val']}")
    print(f" - Test Images         : {split_counts['test']}")
    print("-" * 65)

    if missing_matches > 0:
        print(f"[CRITICAL ERROR] {missing_matches} images failed to match a subject split!")
    if missing_source_files > 0:
        print(f"[CRITICAL ERROR] {missing_source_files} source files were missing!")

    # Per-action class image breakdown
    print("\nPer-Action Image Distribution Across Splits (3-Frame Dataset):")
    print(f"{'Action Class':<22} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-" * 65)
    for act in actions_list:
        tr_c = matrix_counts[act]["train"]
        va_c = matrix_counts[act]["val"]
        te_c = matrix_counts[act]["test"]
        tot_c = tr_c + va_c + te_c
        print(f"{act:<22} | {tr_c:<7} | {va_c:<7} | {te_c:<7} | {tot_c:<7}")
    print("-" * 65)
    print(f"{'TOTAL':<22} | {split_counts['train']:<7} | {split_counts['val']:<7} | {split_counts['test']:<7} | {copied_count:<7}")
    print("=" * 65)

    if copied_count == len(extraction_rows) and missing_matches == 0 and missing_source_files == 0:
        print("VERIFICATION SUCCESS: All 3-frame images materialized to data_split_3frame/!")
    else:
        print("[WARNING] Copy count mismatch or errors occurred during materialization.")

if __name__ == "__main__":
    main()
