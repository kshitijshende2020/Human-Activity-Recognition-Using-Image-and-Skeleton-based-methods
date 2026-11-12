import os
import re
import csv
import cv2
from pathlib import Path

def main():
    # Base paths relative to project root (HAR-Image Based/)
    project_root = Path(__file__).resolve().parent.parent
    video_rgb_dir = project_root / "dataset" / "VIDEO_RGB"
    frames_raw_3frame_dir = project_root / "frames_raw_3frame"
    manifests_dir = project_root / "manifests"
    log_csv_path = manifests_dir / "extraction_log_3frame.csv"

    # Ensure output directories exist
    frames_raw_3frame_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Exact regex pattern: p(\d+)_([a-zA-Z0-9]+)_s(\d+)\.avi
    pattern = re.compile(r"^p(\d+)_([a-zA-Z0-9]+)_s(\d+)\.avi$", re.IGNORECASE)

    log_rows = []
    processed_count = 0
    written_count = 0
    skipped_files = []

    action_dirs = sorted([d for d in video_rgb_dir.iterdir() if d.is_dir()])

    print("=" * 65)
    print("Stage 1 (3-Frame Pipeline) — 3-Frame Extraction Starting")
    print(f"Input Directory  : {video_rgb_dir}")
    print(f"Output Directory : {frames_raw_3frame_dir}")
    print(f"Log CSV Path     : {log_csv_path}")
    print("=" * 65)

    # Relative frame position configurations: (pct, suffix, position_label)
    positions = [
        (0.25, "frame1", "25pct"),
        (0.50, "frame2", "50pct"),
        (0.75, "frame3", "75pct")
    ]

    for action_dir in action_dirs:
        action_name = action_dir.name  # e.g., 'backhand', 'flat_service'
        out_action_dir = frames_raw_3frame_dir / action_name
        out_action_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(list(action_dir.glob("*.avi")))

        for video_path in video_files:
            processed_count += 1
            filename = video_path.name

            match = pattern.match(filename)
            if not match:
                print(f"[WARNING] Filename '{filename}' does not match pattern. Skipping.")
                skipped_files.append((str(video_path), "Regex pattern mismatch"))
                continue

            subject_num, action_code, take_num = match.groups()
            subject_id = f"p{subject_num}"

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[ERROR] Could not open video: {video_path}")
                skipped_files.append((str(video_path), "Failed to open video"))
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"[WARNING] Invalid frame count ({total_frames}) for: {video_path}")
                cap.release()
                skipped_files.append((str(video_path), "Invalid frame count"))
                continue

            # Extract 3 frames per video clip
            for rel_pct, frame_suffix, pos_label in positions:
                frame_idx = min(max(0, int(total_frames * rel_pct)), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[ERROR] Could not read frame {frame_idx} ({pos_label}) from: {video_path}")
                    skipped_files.append((str(video_path), f"Failed frame read at {pos_label}"))
                    continue

                # Filenames: p{subjectID}_{action}_s{take}_frame1.jpg, frame2.jpg, frame3.jpg
                out_frame_name = f"{subject_id}_{action_code}_s{take_num}_{frame_suffix}.jpg"
                out_frame_path = out_action_dir / out_frame_name

                success = cv2.imwrite(str(out_frame_path), frame)
                if not success:
                    print(f"[ERROR] Failed to save image: {out_frame_path}")
                    skipped_files.append((str(video_path), f"Failed write at {pos_label}"))
                    continue

                written_count += 1

                rel_source_video = video_path.relative_to(project_root).as_posix()
                rel_output_image = out_frame_path.relative_to(project_root).as_posix()

                log_rows.append({
                    "subject_id": subject_id,
                    "action": action_name,
                    "take": take_num,
                    "source_video": rel_source_video,
                    "output_image": rel_output_image,
                    "frame_position": pos_label
                })

            cap.release()

    # Write log CSV for 3-frame dataset
    fieldnames = ["subject_id", "action", "take", "source_video", "output_image", "frame_position"]
    with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print("\n" + "=" * 65)
    print("STAGE 1 (3-FRAME PIPELINE) EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Total videos processed : {processed_count}")
    print(f"Total images written   : {written_count} (Expected ~5,940)")
    print(f"Total skipped entries  : {len(skipped_files)}")
    if skipped_files:
        print("\nSkipped Files Details:")
        for path, reason in skipped_files:
            print(f"  - {path}: {reason}")
    print(f"Manifest saved to      : {log_csv_path.as_posix()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
