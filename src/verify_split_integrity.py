import csv
import re
from pathlib import Path

def get_subject_num(subject_id: str) -> int:
    """Extract integer subject number for sorting (e.g. 'p12' -> 12)."""
    match = re.search(r"\d+", subject_id)
    return int(match.group(0)) if match else 9999

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_split_dir = project_root / "data_split"
    manifest_path = project_root / "manifests" / "split_manifest.csv"

    if not data_split_dir.exists():
        raise FileNotFoundError(f"data_split directory not found at: {data_split_dir}")

    # Regex to extract subject ID directly from image filename (e.g. p10_backhand_s1_frame1.jpg -> p10)
    subj_pattern = re.compile(r"^(p\d+)", re.IGNORECASE)

    split_subjects = {
        "train": set(),
        "val": set(),
        "test": set()
    }

    # Maps subject_id -> set of splits where images for that subject are located on disk
    subject_disk_map = {}

    print("=" * 70)
    print("INDEPENDENT SPLIT INTEGRITY VERIFICATION")
    print(f"Scanning directory: {data_split_dir}")
    print("=" * 70)

    total_images_scanned = 0

    for split in ["train", "val", "test"]:
        split_path = data_split_dir / split
        if not split_path.exists():
            print(f"[WARNING] Split folder missing: {split_path}")
            continue

        for jpg_file in split_path.glob("*/*.jpg"):
            total_images_scanned += 1
            filename = jpg_file.name
            match = subj_pattern.match(filename)

            if not match:
                print(f"[ERROR] Unable to parse subject_id from filename: {jpg_file}")
                continue

            subj_id = match.group(1).lower()
            split_subjects[split].add(subj_id)
            
            if subj_id not in subject_disk_map:
                subject_disk_map[subj_id] = set()
            subject_disk_map[subj_id].add(split)

    # 1. Print visual lists of subjects per split
    print(f"\nTotal Images Scanned on Disk: {total_images_scanned}\n")
    for split in ["train", "val", "test"]:
        sorted_subjs = sorted(list(split_subjects[split]), key=get_subject_num)
        print(f"[{split.upper()} SPLIT] Total Subjects: {len(sorted_subjs)}")
        print(f"  {', '.join(sorted_subjs)}\n")

    # 2. Check Set Intersections for Subject Data Leakage
    train_val_overlap = split_subjects["train"] & split_subjects["val"]
    train_test_overlap = split_subjects["train"] & split_subjects["test"]
    val_test_overlap = split_subjects["val"] & split_subjects["test"]

    all_disk_subjects = split_subjects["train"] | split_subjects["val"] | split_subjects["test"]
    sorted_all_subjects = sorted(list(all_disk_subjects), key=get_subject_num)

    has_leakage = False

    print("-" * 70)
    print("INTERSECTION & LEAKAGE CHECKS:")
    print("-" * 70)

    if train_val_overlap:
        has_leakage = True
        print(f"❌ [FAIL] TRAIN ∩ VAL LEAKAGE DETECTED!")
        print(f"   Leaking subjects: {sorted(list(train_val_overlap), key=get_subject_num)}")
    else:
        print("✅ Train ∩ Val   : EMPTY (No overlap)")

    if train_test_overlap:
        has_leakage = True
        print(f"❌ [FAIL] TRAIN ∩ TEST LEAKAGE DETECTED!")
        print(f"   Leaking subjects: {sorted(list(train_test_overlap), key=get_subject_num)}")
    else:
        print("✅ Train ∩ Test  : EMPTY (No overlap)")

    if val_test_overlap:
        has_leakage = True
        print(f"❌ [FAIL] VAL ∩ TEST LEAKAGE DETECTED!")
        print(f"   Leaking subjects: {sorted(list(val_test_overlap), key=get_subject_num)}")
    else:
        print("✅ Val ∩ Test    : EMPTY (No overlap)")

    print(f"\nTotal Unique Subjects Found on Disk: {len(all_disk_subjects)} (Expected: 55)")

    # 3. Secondary Cross-Check against split_manifest.csv
    manifest_mismatch = False
    if manifest_path.exists():
        print("\n" + "-" * 70)
        print("CROSS-CHECKING DISK FILES AGAINST MANIFEST (split_manifest.csv):")
        print("-" * 70)
        manifest_map = {}
        with open(manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_map[row["subject_id"].strip().lower()] = row["split"].strip().lower()

        mismatches = []
        for subj_id in sorted_all_subjects:
            disk_splits = subject_disk_map.get(subj_id, set())
            expected_split = manifest_map.get(subj_id)

            if expected_split is None:
                mismatches.append(f"Subject '{subj_id}' on disk but missing from manifest CSV!")
            elif disk_splits != {expected_split}:
                mismatches.append(
                    f"Subject '{subj_id}' expected in '{expected_split}' per manifest, "
                    f"but found in {sorted(list(disk_splits))} on disk!"
                )

        if mismatches:
            manifest_mismatch = True
            print("❌ Manifest Mismatches Found:")
            for m in mismatches:
                print(f"   - {m}")
        else:
            print("✅ All subject folder locations on disk match split_manifest.csv perfectly!")
    else:
        print(f"[WARNING] split_manifest.csv not found at: {manifest_path}")

    # 4. Final Verdict Output
    print("\n" + "=" * 70)
    print("VERIFICATION FINAL VERDICT:")
    print("=" * 70)

    if not has_leakage and not manifest_mismatch and len(all_disk_subjects) == 55:
        print("PASS — SPLITS ARE FULLY DISJOINT")
    else:
        print("FAIL — LEAKAGE DETECTED")
        if has_leakage:
            print(" -> Data leakage detected across train/val/test splits.")
        if manifest_mismatch:
            print(" -> Folder mismatch detected between disk and split_manifest.csv.")
        if len(all_disk_subjects) != 55:
            print(f" -> Expected 55 subjects, but found {len(all_disk_subjects)}.")
    print("=" * 70)

if __name__ == "__main__":
    main()
