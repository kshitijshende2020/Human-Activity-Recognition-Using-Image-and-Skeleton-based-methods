# =====================================================================
# PHASE 2c - CLEAN + NORMALIZE + RESAMPLE (paste into ONE Colab cell)
# Reads  : /content/skeleton_data   (originals, after the retracked fixes were copied in)
# Writes : /content/skeleton_clean  (new folder; originals are never modified)
# Rules are PROVISIONAL - thresholds are all at the top and the report shows what each rule touched.
# =====================================================================
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SKEL_ROOT  = "/content/skeleton_data"
CLEAN_ROOT = "/content/skeleton_clean"
N_OUT        = 64      # every video is resampled to this many time steps
CONF         = 0.5     # a joint is "usable" above this confidence
KEY          = [5, 6, 7, 8, 9, 10, 11, 12]   # shoulders, elbows, wrists, hips
MIN_KEY_OK   = 6       # a frame needs at least 6 of these 8 joints confident
HEIGHT_RATIO = 1.5     # frame is bad if person-box height differs from its neighbours' median by more than this factor
HIP_SPIKE    = 0.25    # frame is bad if hip centre is this many body-heights away from its neighbours' midpoint
MED_WIN      = 9       # window (frames) for the neighbours' median
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def clean_video(path):
    d = np.load(path)
    k, c, b = d["kpts"].copy(), d["conf"], d["box"]
    fps = float(d["fps"]); T = len(k)

    # ---- 1. which frames are reliable? ----
    has = ~np.isnan(b[:, 0])
    key_ok = ((c[:, KEY] > CONF).sum(1) >= MIN_KEY_OK) & has
    h = b[:, 3] - b[:, 1]
    med_h = pd.Series(h).rolling(MED_WIN, center=True, min_periods=1).median().values
    with np.errstate(invalid="ignore", divide="ignore"):
        r = h / med_h
        size_ok = np.isfinite(r) & (np.maximum(r, 1 / r) <= HEIGHT_RATIO)
    hip = (k[:, 11] + k[:, 12]) / 2
    body_h = np.nanmedian(h) if has.any() else np.nan
    spike = np.zeros(T, bool)
    for t in range(1, T - 1):
        if key_ok[t - 1] and key_ok[t] and key_ok[t + 1]:
            spike[t] = np.linalg.norm(hip[t] - (hip[t - 1] + hip[t + 1]) / 2) / body_h > HIP_SPIKE
    good = key_ok & size_ok & ~spike

    # ---- 2. replace unusable joint values by interpolating over time ----
    jvalid = good[:, None] & (c > CONF)
    if (jvalid[:, [5, 6, 11, 12]].sum(0) == 0).any():
        return None                                   # no usable shoulders/hips at all (should not happen)
    t_idx = np.arange(T)
    for j in range(17):
        v = jvalid[:, j]
        if v.sum() == 0:
            k[:, j, :] = np.nan
            continue
        for ax in (0, 1):
            k[:, j, ax] = np.interp(t_idx, t_idx[v], k[v, j, ax])

    # ---- 3. normalise: centre on hips, scale by this video's own torso length ----
    hip_c = (k[:, 11] + k[:, 12]) / 2
    sh_c = (k[:, 5] + k[:, 6]) / 2
    for j in range(17):
        if np.isnan(k[:, j, 0]).any():
            k[:, j, :] = hip_c                        # joint never usable: put it at the hips
    scale = float(np.median(np.linalg.norm(sh_c - hip_c, axis=1)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    seq = (k - hip_c[:, None, :]) / scale

    # ---- 4. resample to a fixed number of time steps ----
    t_old, t_new = np.linspace(0, 1, T), np.linspace(0, 1, N_OUT)
    out = np.stack([np.stack([np.interp(t_new, t_old, seq[:, j, a]) for a in (0, 1)], -1)
                    for j in range(17)], 1).astype(np.float32)             # (N_OUT, 17, 2)
    mid = np.array([[np.interp(0.5, t_old, seq[:, j, a]) for a in (0, 1)]
                    for j in range(17)], np.float32)                       # pose at exactly 50% of the swing
    stats = dict(n_frames=T, fps=fps, scale_px=scale, frac_bad=float((~good).mean()),
                 n_bad=int((~good).sum()), frac_interp=float(1 - jvalid.mean()),
                 bad_noplayer=int((~has).sum()), bad_joints=int((~key_ok & has).sum()),
                 bad_size=int((~size_ok & has).sum()), bad_spike=int(spike.sum()))
    return out, mid, stats

files = sorted(p for p in glob.glob(f"{SKEL_ROOT}/*/*.npz") if not p.endswith(".tmp.npz"))
rows, out_paths = [], []
for p in files:
    cls, stem = os.path.basename(os.path.dirname(p)), os.path.splitext(os.path.basename(p))[0]
    res = clean_video(p)
    if res is None:
        rows.append(dict(stem=stem, cls=cls, failed=True)); out_paths.append(None); continue
    out, mid, st = res
    os.makedirs(f"{CLEAN_ROOT}/{cls}", exist_ok=True)
    op = f"{CLEAN_ROOT}/{cls}/{stem}.npz"
    np.savez_compressed(op, seq=out, mid_pose=mid, n_frames=st["n_frames"], fps=st["fps"],
                        duration=st["n_frames"] / st["fps"], scale_px=st["scale_px"], frac_bad=st["frac_bad"])
    assert np.isfinite(out).all() and np.isfinite(mid).all(), f"non-finite values in {stem}"
    rows.append(dict(stem=stem, cls=cls, failed=False, **st)); out_paths.append(op)

rep = pd.DataFrame(rows)
rep.to_csv("/content/skeleton_clean_report.csv", index=False)
ok = rep[~rep.failed]
print(f"Cleaned videos written : {len(ok)} / {len(files)}   (failed: {int(rep.failed.sum())})")
print(f"Videos with >=1 bad frame : {(ok.n_bad > 0).sum()} | >10% bad: {(ok.frac_bad > 0.10).sum()} | >25% bad: {(ok.frac_bad > 0.25).sum()}")
print("\nWhy frames were marked bad (total frames over all videos):")
print(ok[["bad_noplayer", "bad_joints", "bad_size", "bad_spike"]].sum().to_string())
print("\nFraction of bad frames per video:")
print(ok.frac_bad.describe(percentiles=[.5, .9, .95, .99]).round(3).to_string())
print("\nFraction of joint values replaced by interpolation, per video:")
print(ok.frac_interp.describe(percentiles=[.5, .9, .95, .99]).round(3).to_string())
print("\n15 videos with the most bad frames:")
print(ok.sort_values("frac_bad", ascending=False).head(15)[["stem", "frac_bad", "n_bad", "n_frames"]].to_string(index=False))
print("\nMean fraction bad frames by class:")
print(ok.groupby("cls").frac_bad.mean().round(4).to_string())

# quick visual sanity check: 3 random cleaned videos, 8 time steps each (torso length = 1 unit)
valid = [i for i, p in enumerate(out_paths) if p]
pick = np.random.default_rng(0).choice(valid, 3, replace=False)
fig, ax = plt.subplots(3, 8, figsize=(16, 8))
for r_, i in enumerate(pick):
    seq = np.load(out_paths[i])["seq"]
    for c_, t in enumerate(np.linspace(0, N_OUT - 1, 8).astype(int)):
        a = ax[r_, c_]
        for u, v in EDGES:
            a.plot(seq[t, [u, v], 0], seq[t, [u, v], 1], "g-")
        a.set_xlim(-2.5, 2.5); a.set_ylim(3.5, -2.0); a.set_aspect("equal"); a.axis("off")
        if c_ == 0:
            a.set_title(os.path.basename(out_paths[i])[:-4], fontsize=9, loc="left")
plt.tight_layout(); plt.show()
