# =====================================================================
# PHASE 1 - SKELETON EXTRACTION (paste each CELL into its own Colab cell)
# Run CELL 1 first. Run CELL 2 only if CELL 1 prints "ALL CHECKS PASSED".
# =====================================================================


# ============ CELL 1: setup + inventory + consistency checks ============
import os, re, glob, json
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')

PROJECT    = "/content/drive/MyDrive/HAR Image Based/HAR-Image Based"
VIDEO_ROOT = "/content/thetis_rgb"
MANIFEST   = f"{PROJECT}/manifests/split_manifest.csv"
OUT_ROOT   = f"{PROJECT}/skeleton_data"

RAW_TO_CLASS = {
    "backhand": "backhand", "bslice": "backhand_slice", "serflat": "flat_service",
    "serslice": "slice_service", "foreflat": "forehand_flat", "fslice": "forehand_slice",
    "backhand2h": "backhand2hands", "bvolley": "backhand_volley", "serkick": "kick_service",
    "smash": "smash", "foreopen": "forehand_openstands", "fvolley": "forehand_volley",
}

# ---- load the SAME split manifest used for the pixel experiments ----
man = pd.read_csv(MANIFEST)
print("Manifest columns:", man.columns.tolist())
print(man.head(3), "\n")

subj_col  = next(c for c in man.columns if "subject" in c.lower())
split_col = next(c for c in man.columns if "split" in c.lower())
print(f"Using subject column = '{subj_col}', split column = '{split_col}'")

def norm_subj(s):
    s = str(s).strip().lower()
    return s if s.startswith("p") else f"p{s}"

man["_subj"] = man[subj_col].map(norm_subj)
assert man.groupby("_subj")[split_col].nunique().max() == 1, "A subject appears in more than one split!"
subj2split = man.drop_duplicates("_subj").set_index("_subj")[split_col].to_dict()

# ---- inventory of the raw videos ----
videos = sorted(glob.glob(f"{VIDEO_ROOT}/**/*.avi", recursive=True))
records, bad = [], []
for v in videos:
    stem = os.path.splitext(os.path.basename(v))[0]
    parts = stem.split("_")
    m = re.fullmatch(r"s(\d+)", parts[-1])
    if len(parts) < 3 or not m:
        bad.append(v); continue
    subj = parts[0].lower()
    raw = "_".join(parts[1:-1])
    cls = RAW_TO_CLASS.get(raw)
    if cls is None or subj not in subj2split:
        bad.append(v); continue
    records.append(dict(path=v, stem=stem, subject=subj, cls=cls,
                        take=int(m.group(1)), split=subj2split[subj]))
inv = pd.DataFrame(records)

print("\nVideos found           :", len(videos))
print("Videos recognised      :", len(inv))
print("Unrecognised filenames :", len(bad), bad[:5])
print("Subjects per split     :", inv.drop_duplicates("subject").groupby("split").size().to_dict())
print("Videos per split       :", inv.groupby("split").size().to_dict())
print("Videos per class       :", inv.groupby("cls").size().to_dict())

# ---- consistency checks (must all pass before extraction) ----
assert not bad, "Some filenames could not be parsed/mapped - paste the list above"
assert len(inv) == 1980, f"Expected 1980 videos, got {len(inv)}"
assert sorted(inv.groupby("split").size().tolist()) == [288, 288, 1404], "Split sizes differ from the pixel experiments"
assert (inv.groupby("cls").size() == 165).all(), "Each class should have 165 videos (55 subjects x 3 takes)"
print("\nALL CHECKS PASSED")


# ============ CELL 2: extract skeletons for every frame (resumable) ============
import time
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

assert len(inv) == 1980, "Run CELL 1 first and make sure all checks passed"

MODEL_NAME = "yolo11m-pose.pt"   # same model + settings as the Phase 0 test
IMG_SIZE   = 640
CHUNK      = 16                  # frames per inference call

os.makedirs(OUT_ROOT, exist_ok=True)
cfg = dict(model=MODEL_NAME, imgsz=IMG_SIZE, person_rule="largest bounding box per frame",
           keypoints="COCO-17 (x, y in pixels) + per-joint confidence",
           missing_frames="kpts = NaN, conf = 0", ultralytics=ultralytics.__version__)
with open(f"{OUT_ROOT}/extraction_config.json", "w") as f:
    json.dump(cfg, f, indent=2)

device = 0 if torch.cuda.is_available() else "cpu"
print("Device:", "GPU" if device == 0 else "CPU (very slow - switch runtime to T4 GPU)")
model = YOLO(MODEL_NAME)

def read_all_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    h, w = frames[0].shape[:2] if frames else (0, 0)
    return frames, fps, w, h

def extract(frames):
    T = len(frames)
    kpts  = np.full((T, 17, 2), np.nan, np.float32)
    conf  = np.zeros((T, 17), np.float32)
    box   = np.full((T, 4), np.nan, np.float32)
    npers = np.zeros(T, np.int16)
    for i in range(0, T, CHUNK):
        res = model(frames[i:i + CHUNK], imgsz=IMG_SIZE, device=device, verbose=False)
        for j, r in enumerate(res):
            t = i + j
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                continue
            b = r.boxes.xyxy.cpu().numpy()
            npers[t] = len(b)
            k = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))  # largest person
            kpts[t] = r.keypoints.xy[k].cpu().numpy()
            if r.keypoints.conf is not None:
                conf[t] = r.keypoints.conf[k].cpu().numpy()
            box[t] = b[k]
    return kpts, conf, box, npers

t_start = time.time()
done = skipped = 0
failed = []

for n, row in enumerate(inv.itertuples(), 1):
    out_dir = f"{OUT_ROOT}/{row.cls}"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{row.stem}.npz"
    if os.path.exists(out):          # resume: skip videos already extracted
        skipped += 1
        continue
    try:
        frames, fps, w, h = read_all_frames(row.path)
        if not frames:
            raise RuntimeError("no frames could be read")
        kpts, conf, box, npers = extract(frames)
        tmp = out + ".tmp.npz"       # write to temp name, then rename (no half-written files)
        np.savez_compressed(tmp, kpts=kpts, conf=conf, box=box, n_persons=npers,
                            fps=np.float32(fps), width=np.int32(w), height=np.int32(h))
        os.replace(tmp, out)
        done += 1
    except Exception as e:
        failed.append((row.stem, str(e)))

    if n % 25 == 0:
        el = time.time() - t_start
        rate = el / max(done, 1)
        remaining = len(inv) - n
        print(f"[{n}/{len(inv)}] extracted={done} skipped={skipped} failed={len(failed)} "
              f"| elapsed {el/60:.1f} min | est. remaining {rate*remaining/60:.1f} min")

with open(f"{OUT_ROOT}/extraction_failures.json", "w") as f:
    json.dump(failed, f, indent=2)

n_files = len([p for p in glob.glob(f"{OUT_ROOT}/*/*.npz") if not p.endswith(".tmp.npz")])
print(f"\nFINISHED | extracted now: {done} | already existed: {skipped} | failed: {len(failed)}")
print(f"Skeleton files on disk: {n_files} / 1980")
if failed:
    print("Failed videos (first 10):", failed[:10])
