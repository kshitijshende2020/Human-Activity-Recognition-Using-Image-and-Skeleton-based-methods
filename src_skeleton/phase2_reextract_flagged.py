# =====================================================================
# PHASE 2b - RE-EXTRACT THE FLAGGED VIDEOS WITH "FOLLOW ONE PERSON" RULE
# Paste into ONE Colab cell. Needs (still in /content): thetis_rgb/, skeleton_data/,
# skeleton_qc.csv. Writes ONLY to /content/skeleton_data_retracked (originals untouched).
# =====================================================================
import os, glob
import numpy as np, pandas as pd, cv2, torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

SKEL_ROOT  = "/content/skeleton_data"
NEW_ROOT   = "/content/skeleton_data_retracked"
VIDEO_ROOT = "/content/thetis_rgb"
QC_CSV     = "/content/skeleton_qc.csv"
MODEL_NAME, IMG_SIZE, CHUNK = "yolo11m-pose.pt", 640, 16     # same model/settings as before
EXTRA_STEMS = []        # optionally add more video names (without .avi) to redo
CONF = 0.5
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

qc = pd.read_csv(QC_CSV)
sel = qc[(qc.max_jump > 0.2) | (qc.max_size_ratio > 1.5)]
targets = sorted(set(sel.stem) | set(EXTRA_STEMS))
print(len(targets), "videos to re-extract:", targets)

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_NAME)

def read_frames(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    return fr, fps, fr[0].shape[1], fr[0].shape[0]

def detect_all(frames):
    """Keep EVERY detected person per frame: list of (box, kpts, conf)."""
    dets = []
    for i in range(0, len(frames), CHUNK):
        for r in model(frames[i:i + CHUNK], imgsz=IMG_SIZE, device=device, verbose=False):
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                dets.append([]); continue
            b = r.boxes.xyxy.cpu().numpy(); xy = r.keypoints.xy.cpu().numpy()
            cf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.zeros(xy.shape[:2])
            dets.append([(b[k], xy[k], cf[k]) for k in range(len(b))])
    return dets

centre = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
height = lambda b: b[3] - b[1]
area   = lambda b: (b[2] - b[0]) * (b[3] - b[1])

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0

def follow_player(dets):
    """Pick the player once (large + near the video's typical player position),
    then follow the SAME person frame to frame by box overlap/closeness."""
    T = len(dets)
    big = [max(dd, key=lambda d: area(d[0])) for dd in dets if dd]
    if not big:
        return [None] * T
    ref = np.median([centre(d[0]) for d in big], axis=0)
    ref_h = np.median([height(d[0]) for d in big])
    best, start = None, None
    for t, dd in enumerate(dets):
        for k, d in enumerate(dd):
            if height(d[0]) < 0.6 * ref_h:
                continue
            dist = np.linalg.norm(centre(d[0]) - ref)
            if best is None or dist < best:
                best, start = dist, (t, k)
    chosen = [None] * T
    t0, k0 = start
    chosen[t0] = dets[t0][k0]
    for order in (range(t0 + 1, T), range(t0 - 1, -1, -1)):
        prev = chosen[t0]
        for t in order:
            cand = dets[t]
            if not cand:
                continue
            ious = [iou(prev[0], d[0]) for d in cand]
            dists = [np.linalg.norm(centre(d[0]) - centre(prev[0])) for d in cand]
            k = int(np.argmax(ious))
            if ious[k] <= 0.1:
                k = int(np.argmin(dists))
            if ious[k] > 0.1 or dists[k] < 0.5 * height(prev[0]):
                chosen[t] = cand[k]; prev = cand[k]      # else: leave frame empty
    return chosen

def to_arrays(chosen, dets):
    T = len(chosen)
    kpts = np.full((T, 17, 2), np.nan, np.float32); conf = np.zeros((T, 17), np.float32)
    box = np.full((T, 4), np.nan, np.float32); npers = np.array([len(d) for d in dets], np.int16)
    for t, c in enumerate(chosen):
        if c is not None:
            box[t], kpts[t], conf[t] = c[0], c[1], c[2]
    return kpts, conf, box, npers

def stats(kpts, conf, box):
    hipok = (conf[:, 11] > CONF) & (conf[:, 12] > CONF); hip = (kpts[:, 11] + kpts[:, 12]) / 2
    h = box[:, 3] - box[:, 1]; med = np.nanmedian(h); idx = np.where(hipok)[0]
    jump = float((np.linalg.norm(hip[idx[1:]] - hip[idx[:-1]], axis=1) / med / np.diff(idx)).max()) if len(idx) > 1 else np.nan
    ok = ~np.isnan(h); di = np.where(ok[1:] & ok[:-1])[0]
    r = h[di + 1] / h[di] if len(di) else np.array([1.0])
    return jump, float(np.max(np.maximum(r, 1 / r)))

def draw(fr, xy, cf):
    img = fr.copy()
    for a, b in EDGES:
        if cf[a] > CONF and cf[b] > CONF:
            cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), (0, 255, 0), 2)
    return img

report = []
for stem in targets:
    row = qc[qc.stem == stem].iloc[0]
    hits = glob.glob(f"{VIDEO_ROOT}/**/{stem}.avi", recursive=True)
    if not hits:
        print("raw video not found:", stem); continue
    frames, fps, w, h = read_frames(hits[0])
    dets = detect_all(frames)
    chosen = follow_player(dets)
    kpts, conf, box, npers = to_arrays(chosen, dets)
    os.makedirs(f"{NEW_ROOT}/{row.cls}", exist_ok=True)
    np.savez_compressed(f"{NEW_ROOT}/{row.cls}/{stem}.npz", kpts=kpts, conf=conf, box=box,
                        n_persons=npers, fps=np.float32(fps), width=np.int32(w), height=np.int32(h))
    nj, ns = stats(kpts, conf, box)
    report.append(dict(stem=stem, old_jump=round(row.max_jump, 3), new_jump=round(nj, 3),
                       old_size=round(row.max_size_ratio, 3), new_size=round(ns, 3),
                       frames_without_player=int(np.isnan(box[:, 0]).sum()), n_frames=len(frames)))
    # visual check: the old problem frames + the middle of the clip, NEW skeleton drawn
    pick = [int(row.jump_prev), int(row.jump_cur), len(frames) // 2] if row.jump_cur >= 0 else [0, len(frames) // 2, len(frames) - 1]
    tiles = [draw(frames[i], kpts[i], conf[i]) for i in pick]
    plt.figure(figsize=(15, 4)); plt.imshow(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB))
    plt.title(f"{stem} (NEW tracking) - frames {pick}"); plt.axis("off"); plt.show()

print(pd.DataFrame(report).to_string(index=False))

# ---------------------------------------------------------------------
# ONLY after you have looked at every picture above and confirmed the skeleton
# follows the player, run this to replace the old files (keep your downloaded
# skeleton_data zip as the backup of the originals):
#
# import shutil
# for p in glob.glob(f"{NEW_ROOT}/*/*.npz"):
#     shutil.copy(p, p.replace(NEW_ROOT, SKEL_ROOT))
# ---------------------------------------------------------------------
