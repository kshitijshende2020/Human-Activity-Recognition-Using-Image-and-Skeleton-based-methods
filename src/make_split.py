import csv
import random
import re
from pathlib import Path

def get_subject_number(subject_id: str) -> int:
    """Extract integer subject number from string like 'p12' -> 12."""
    match = re.search(r"\d+", subject_id)
    if match:
        return int(match.group(0))
    raise ValueError(f"Invalid subject_id format: '{subject_id}'")

def main():
    project_root = Path(__file__).resolve().parent.parent
    manifests_dir = project_root / "manifests"
    log_csv_path = manifests_dir / "extraction_log.csv"
    split_csv_path = manifests_dir / "split_manifest.csv"

    if not log_csv_path.exists():
        raise FileNotFoundError(f"Input extraction log not found: {log_csv_path}")

    # 1. Read unique subject IDs from extraction_log.csv
    unique_subjects = set()
    with open(log_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_subjects.add(row["subject_id"].strip())

    # Sort numerically by subject number (p1, p2, ..., p55)
    subject_list = sorted(list(unique_subjects), key=get_subject_number)

    print("=" * 60)
    print("Stage 2 — Subject-Wise Train/Val/Test Split Starting")
    print(f"Total Unique Subjects Found: {len(subject_list)}")
    print("=" * 60)

    # 2. Classify subjects into Beginners (p1–p31) and Experts (p32–p55)
    beginners = []
    experts = []

    for subj in subject_list:
        num = get_subject_number(subj)
        if 1 <= num <= 31:
            beginners.append(subj)
        elif 32 <= num <= 55:
            experts.append(subj)
        else:
            print(f"[WARNING] Subject '{subj}' (number {num}) outside expected p1-p55 range.")

    print(f"Beginner Subjects (p1–p31) : {len(beginners)}")
    print(f"Expert Subjects   (p32–p55) : {len(experts)}")

    # 3. Perform stratified split with fixed random seed (42)
    RANDOM_SEED = 42
    rng = random.Random(RANDOM_SEED)

    shuffled_beginners = beginners.copy()
    shuffled_experts = experts.copy()

    # Shuffle beginner and expert lists deterministically
    rng.shuffle(shuffled_beginners)
    rng.shuffle(shuffled_experts)

    # Beginner split (31 subjects): 22 train (~71%), 4 val (~13%), 5 test (~16%)
    b_train = shuffled_beginners[:22]
    b_val = shuffled_beginners[22:26]
    b_test = shuffled_beginners[26:]

    # Expert split (24 subjects): 17 train (~71%), 4 val (~17%), 3 test (~12.5%)
    e_train = shuffled_experts[:17]
    e_val = shuffled_experts[17:21]
    e_test = shuffled_experts[21:]

    # Combine split groups
    train_set = set(b_train + e_train)
    val_set = set(b_val + e_val)
    test_set = set(b_test + e_test)

    # 4. Strict Sanity Check Assertions
    assert len(train_set & val_set) == 0, "CRITICAL ERROR: Overlap between Train and Val sets!"
    assert len(train_set & test_set) == 0, "CRITICAL ERROR: Overlap between Train and Test sets!"
    assert len(val_set & test_set) == 0, "CRITICAL ERROR: Overlap between Val and Test sets!"

    all_assigned = train_set | val_set | test_set
    assert len(all_assigned) == len(subject_list), "CRITICAL ERROR: Not all subjects assigned to splits!"

    # 5. Build and write split_manifest.csv
    split_rows = []
    for subj in subject_list:
        if subj in train_set:
            s_val = "train"
        elif subj in val_set:
            s_val = "val"
        elif subj in test_set:
            s_val = "test"
        else:
            raise ValueError(f"Subject '{subj}' not assigned to any split.")

        split_rows.append({"subject_id": subj, "split": s_val})

    with open(split_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "split"])
        writer.writeheader()
        writer.writerows(split_rows)

    # 6. Print detailed summary
    print("\n" + "=" * 60)
    print(f"STAGE 2 SUBJECT SPLIT COMPLETE (Random Seed = {RANDOM_SEED})")
    print("=" * 60)
    print(f"Total Subjects    : {len(subject_list)}")
    print(f" - Train Split    : {len(train_set)} subjects ({len(b_train)} Beginners, {len(e_train)} Experts) ~70.9%")
    print(f" - Val Split      : {len(val_set)} subjects ({len(b_val)} Beginners, {len(e_val)} Experts) ~14.5%")
    print(f" - Test Split     : {len(test_set)} subjects ({len(b_test)} Beginners, {len(e_test)} Experts) ~14.5%")
    print("-" * 60)
    print("SANITY CHECK PASSED: Every subject belongs to exactly ONE split.")
    print(f"Split Manifest saved to: {split_csv_path.as_posix()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
