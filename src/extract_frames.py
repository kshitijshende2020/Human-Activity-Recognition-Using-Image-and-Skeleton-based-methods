import os
import re
import csv
import cv2
from pathlib import Path

def main():
    # Base paths relative to project root (HAR-Image Based/)
    project_root = Path(__file__).resolve().parent.parent
    video_rgb_dir = project_root / "dataset" / "VIDEO_RGB"
    frames_raw_dir = project_root / "frames_raw"
    manifests_dir = project_root / "manifests"
    log_csv_path = manifests_dir / "extraction_log.csv"

    # Ensure output directories exist
    frames_raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Exact regex pattern specified: p(\d+)_([a-zA-Z0-9]+)_s(\d+)\.avi
    pattern = re.compile(r"^p(\d+)_([a-zA-Z0-9]+)_s(\d+)\.avi$", re.IGNORECASE)

    log_rows = []
    processed_count = 0
    written_count = 0
    skipped_files = []

    # Iterate over the 12 action subdirectories in VIDEO_RGB
    action_dirs = sorted([d for d in video_rgb_dir.iterdir() if d.is_dir()])
    
    print("=" * 60)
    print("Stage 1 — Frame Extraction Starting")
    print(f"Input Directory  : {video_rgb_dir}")
    print(f"Output Directory : {frames_raw_dir}")
    print(f"Log CSV Path     : {log_csv_path}")
    print("=" * 60)

    for action_dir in action_dirs:
        action_name = action_dir.name  # e.g., 'backhand', 'flat_service'
        out_action_dir = frames_raw_dir / action_name
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

            # Open video using OpenCV
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

            # Target middle frame (~50% position by frame count)
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                print(f"[ERROR] Could not read frame {middle_frame_idx} from: {video_path}")
                skipped_files.append((str(video_path), "Failed frame read"))
                continue

            # Output filename: p{subjectID}_{action}_s{take}_frame1.jpg
            out_frame_name = f"{subject_id}_{action_code}_s{take_num}_frame1.jpg"
            out_frame_path = out_action_dir / out_frame_name

            # Save frame image as JPEG
            success = cv2.imwrite(str(out_frame_path), frame)
            if not success:
                print(f"[ERROR] Failed to save image: {out_frame_path}")
                skipped_files.append((str(video_path), "Failed to write JPEG"))
                continue

            written_count += 1

            # Relative paths for portable manifest logging
            rel_source_video = video_path.relative_to(project_root).as_posix()
            rel_output_image = out_frame_path.relative_to(project_root).as_posix()

            log_rows.append({
                "subject_id": subject_id,
                "action": action_name,
                "take": take_num,
                "source_video": rel_source_video,
                "output_image": rel_output_image
            })

    # Save manifest log CSV
    fieldnames = ["subject_id", "action", "take", "source_video", "output_image"]
    with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print("\n" + "=" * 60)
    print("STAGE 1 FRAME EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total videos processed : {processed_count}")
    print(f"Total images written   : {written_count}")
    print(f"Total files skipped    : {len(skipped_files)}")
    if skipped_files:
        print("\nSkipped Files Details:")
        for path, reason in skipped_files:
            print(f"  - {path}: {reason}")
    print(f"Manifest saved to      : {log_csv_path.as_posix()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
