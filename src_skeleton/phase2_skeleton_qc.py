# =====================================================================
# PHASE 2 - SKELETON QUALITY CHECK (read-only: changes NO data)
# Paste into ONE Colab cell and run. Needs skeleton_data in /content.
# =====================================================================
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

SKEL_ROOT  = "/content/skeleton_data"
MANIFEST   = "/content/split_manifest.csv"   # optional (for per-split table)
VIDEO_ROOT = "/content/thetis_rgb"           # optional (for overlay of suspicious videos)
CONF = 0.5
KEY   = [5, 6, 7, 8, 9, 10, 11, 12]          # shoulders, elbows, wrists, hips
WRIST = [9, 10]
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

# provisional flag thresholds (REPORTED ONLY - nothing is removed or changed here)
FLAG_DET, FLAG_KEY, FLAG_GAP, FLAG_JUMP, FLAG_SIZE = 0.95, 0.80, 5, 0.5, 1.5

def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best

files = sorted(p for p in glob.glob(f"{SKEL_ROOT}/*/*.npz") if not p.endswith(".tmp.npz"))
print("Skeleton files found:", len(files), "(expected 1980)")
fail_path = f"{SKEL_ROOT}/extraction_failures.json"
if os.path.exists(fail_path):
    print("Failures recorded during extraction:", len(json.load(open(fail_path))))

rows = []
for p in files:
    d = np.load(p)
    kpts, conf, box, npers = d["kpts"], d["conf"], d["box"], d["n_persons"]
    T = len(kpts)
    detected = npers > 0
    h = box[:, 3] - box[:, 1]
    med_h = np.nanmedian(h) if detected.any() else np.nan

    # largest sudden hip-centre movement per frame, relative to body height
    hipok = (conf[:, 11] > CONF) & (conf[:, 12] > CONF)
    hip = (kpts[:, 11] + kpts[:, 12]) / 2
    idx = np.where(hipok)[0]
    max_jump, j_prev, j_cur = np.nan, -1, -1
    if len(idx) >= 2 and med_h > 0:
        dist = np.linalg.norm(hip[idx[1:]] - hip[idx[:-1]], axis=1) / med_h / np.diff(idx)
        k = int(np.argmax(dist))
        max_jump, j_prev, j_cur = float(dist[k]), int(idx[k]), int(idx[k + 1])

    # largest sudden change in detected person's box height between adjacent frames
    di = np.where(detected[1:] & detected[:-1] & (h[1:] > 0) & (h[:-1] > 0))[0]
    if len(di):
        r = h[di + 1] / h[di]
        max_size = float(np.max(np.maximum(r, 1 / r)))
    else:
        max_size = np.nan

    stem = os.path.splitext(os.path.basename(p))[0]
    rows.append(dict(stem=stem, cls=os.path.basename(os.path.dirname(p)),
                     subject=stem.split("_")[0].lower(), n_frames=T,
                     det_rate=detected.mean(),
                     key_rate=(conf[:, KEY] > CONF).mean(),
                     wrist_rate=(conf[:, WRIST] > CONF).mean(),
                     max_gap=longest_run(~detected),
                     multi_person=(npers >= 2).mean(),
                     max_jump=max_jump, max_size_ratio=max_size,
                     jump_prev=j_prev, jump_cur=j_cur))

qc = pd.DataFrame(rows)

if os.path.exists(MANIFEST):
    man = pd.read_csv(MANIFEST)
    sc = next(c for c in man.columns if "subject" in c.lower())
    pc = next(c for c in man.columns if "split" in c.lower())
    ns = lambda s: str(s).strip().lower() if str(s).strip().lower().startswith("p") else f"p{str(s).strip().lower()}"
    m = man.assign(_s=man[sc].map(ns)).drop_duplicates("_s").set_index("_s")[pc].to_dict()
    qc["split"] = qc["subject"].map(m)
else:
    qc["split"] = "unknown"

qc.to_csv("/content/skeleton_qc.csv", index=False)
pd.set_option("display.width", 200)

print("\n=== 1. Overall distributions (per video) ===")
cols = ["n_frames", "det_rate", "key_rate", "wrist_rate", "max_gap", "multi_person", "max_jump", "max_size_ratio"]
print(qc[cols].describe(percentiles=[.05, .5, .95]).round(3).T[["min", "5%", "50%", "95%", "max"]])

print("\n=== 2. Provisional flags (nothing removed yet) ===")
flags = {
    f"person found in < {FLAG_DET:.0%} of frames": qc.det_rate < FLAG_DET,
    f"key joints confident in < {FLAG_KEY:.0%} of frames": qc.key_rate < FLAG_KEY,
    f"missing-frame gap longer than {FLAG_GAP}": qc.max_gap > FLAG_GAP,
    f"sudden hip jump > {FLAG_JUMP} body-heights/frame": qc.max_jump > FLAG_JUMP,
    f"sudden body-size change > x{FLAG_SIZE}": qc.max_size_ratio > FLAG_SIZE,
}
for name, mask in flags.items():
    print(f"{int(mask.sum()):5d} videos : {name}")
any_flag = np.any([m.values for m in flags.values()], axis=0)
print(f"{int(any_flag.sum()):5d} videos flagged by at least one rule (of {len(qc)})")
print(f"{(qc.multi_person > 0).mean():.0%} of videos have 2+ people visible in at least one frame")

print("\n=== 3. Mean quality by class (is any stroke systematically worse?) ===")
print(qc.groupby("cls")[["det_rate", "key_rate", "wrist_rate", "n_frames", "max_jump"]].mean().round(3))

print("\n=== 4. Mean quality by split ===")
print(qc.groupby("split")[["det_rate", "key_rate", "wrist_rate", "n_frames"]].mean().round(3))

fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for a, (c, t) in zip(ax.ravel(), [("n_frames", "Frames per video"), ("det_rate", "Person-found rate"),
                                  ("key_rate", "Key-joint confidence rate"), ("max_gap", "Longest missing gap (frames)"),
                                  ("max_jump", "Largest hip jump (body-heights/frame)"), ("max_size_ratio", "Largest body-size change (ratio)")]):
    a.hist(qc[c].dropna(), bins=30); a.set_title(t)
plt.tight_layout(); plt.show()

# overlay the 6 videos with the largest hip jump (only if raw videos are still on disk)
worst = qc.sort_values("max_jump", ascending=False).head(6)
for r in worst.itertuples():
    hits = glob.glob(f"{VIDEO_ROOT}/**/{r.stem}.avi", recursive=True)
    if not hits or r.jump_cur < 0:
        continue
    d = np.load(f"{SKEL_ROOT}/{r.cls}/{r.stem}.npz")
    cap = cv2.VideoCapture(hits[0]); tiles = []
    for fi in (r.jump_prev, r.jump_cur):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
        if not ok:
            continue
        xy, cf = d["kpts"][fi], d["conf"][fi]
        for a, b in EDGES:
            if cf[a] > CONF and cf[b] > CONF:
                cv2.line(fr, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), (0, 255, 0), 2)
        tiles.append(fr)
    cap.release()
    if len(tiles) == 2:
        plt.figure(figsize=(12, 4)); plt.imshow(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB))
        plt.title(f"{r.stem}  jump={r.max_jump:.2f}  (frames {r.jump_prev} -> {r.jump_cur})"); plt.axis("off"); plt.show()

print("\nPer-video table saved to /content/skeleton_qc.csv")
