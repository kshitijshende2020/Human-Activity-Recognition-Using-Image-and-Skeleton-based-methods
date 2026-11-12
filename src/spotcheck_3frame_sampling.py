"""
Temporary Spot-Check Script for 3-Frame Video Sampling Hypothesis (25%, 50%, 75%).

NOTE:
The output directory 'spotcheck_output/' created by this script is a temporary, 
throwaway inspection folder. It can be safely deleted at any time and does not 
affect the main HAR pipeline, 'frames_raw/', 'data_split/', or any manifests.
"""

import cv2
import numpy as np
from pathlib import Path

def create_contact_sheet(frame25, frame50, frame75, banner_height=40):
    """
    Combines 3 OpenCV frames horizontally into a single contact sheet with 
    top text banners identifying 25%, 50%, and 75% frame positions.
    """
    frames = [frame25, frame50, frame75]
    labels = ["25%", "50%", "75%"]
    labeled_frames = []

    h, w, c = frame25.shape

    for frame, label in zip(frames, labels):
        # Resize frame if dimensions vary slightly
        if frame.shape != (h, w, c):
            frame = cv2.resize(frame, (w, h))

        # Create top black banner bar for label text
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)

        # Draw label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (banner_height + text_size[1]) // 2 - 2

        cv2.putText(banner, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Stack banner vertically on top of frame
        labeled_frame = np.vstack([banner, frame])
        labeled_frames.append(labeled_frame)

    # Stitch the 3 labeled frames side-by-side horizontally
    contact_sheet = np.hstack(labeled_frames)
    return contact_sheet

def main():
    project_root = Path(__file__).resolve().parent.parent
    video_rgb_dir = project_root / "dataset" / "VIDEO_RGB"
    out_dir = project_root / "spotcheck_output"

    # Create temporary spotcheck output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_rgb_dir.exists():
        raise FileNotFoundError(f"VIDEO_RGB directory not found at: {video_rgb_dir}")

    action_dirs = sorted([d for d in video_rgb_dir.iterdir() if d.is_dir()])

    total_sampled = 0
    total_created = 0

    print("=" * 65)
    print("Stage Spot-Check — 3-Frame Video Sampling (25%, 50%, 75%)")
    print(f"Input Directory  : {video_rgb_dir}")
    print(f"Output Directory : {out_dir}")
    print("=" * 65)

    for action_dir in action_dirs:
        action_name = action_dir.name
        # Select first 2 videos alphabetically per action folder
        video_files = sorted(list(action_dir.glob("*.avi")))[:2]

        for video_path in video_files:
            total_sampled += 1
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"[ERROR] Could not open video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"[WARNING] Invalid frame count ({total_frames}) for: {video_path}")
                cap.release()
                continue

            # Calculate frame positions at 25%, 50%, and 75% relative frame count
            idx_25 = int(total_frames * 0.25)
            idx_50 = int(total_frames * 0.50)
            idx_75 = int(total_frames * 0.75)

            # Extract 25% frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_25)
            ret25, f25 = cap.read()

            # Extract 50% frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_50)
            ret50, f50 = cap.read()

            # Extract 75% frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_75)
            ret75, f75 = cap.read()

            cap.release()

            if not (ret25 and ret50 and ret75) or f25 is None or f50 is None or f75 is None:
                print(f"[ERROR] Failed to read 3 frames from: {video_path}")
                continue

            # Build side-by-side contact sheet
            contact_sheet = create_contact_sheet(f25, f50, f75)

            # Save to spotcheck_output/<action>_<filename>.jpg
            out_filename = f"{action_name}_{video_path.stem}.jpg"
            out_path = out_dir / out_filename

            cv2.imwrite(str(out_path), contact_sheet)
            total_created += 1
            print(f"[CREATED] {out_path.relative_to(project_root).as_posix()}")

    print("\n" + "=" * 65)
    print("3-FRAME SPOT-CHECK SAMPLING SUMMARY")
    print("=" * 65)
    print(f"Total videos sampled        : {total_sampled}")
    print(f"Total contact sheets created : {total_created}")
    print(f"Output directory            : {out_dir.as_posix()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
