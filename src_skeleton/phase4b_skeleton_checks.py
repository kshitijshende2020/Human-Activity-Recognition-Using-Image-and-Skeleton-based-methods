# =====================================================================
# PHASE 4b - VALIDATION-ONLY CHECKS ON THE SKELETON MODELS (paste into ONE Colab cell)
# Self-contained: needs only /content/skeleton_clean and /content/split_manifest.csv
# The TEST split is never loaded into tensors and never evaluated here.
# =====================================================================
import os, glob, copy, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

CLEAN_ROOT = "/content/skeleton_clean"
MANIFEST   = "/content/split_manifest.csv"
LOG_DIR    = "/content/results_skeleton"
SEEDS = [42, 43, 44, 45, 46]
BATCH, MAX_EPOCHS, PATIENCE, LR, WD = 64, 150, 20, 1e-3, 1e-4
NOISE_STD, SCALE_JIT = 0.02, 0.10
CLASS_NAMES = ["backhand", "backhand2hands", "backhand_slice", "backhand_volley", "flat_service",
               "forehand_flat", "forehand_openstands", "forehand_slice", "forehand_volley",
               "kick_service", "slice_service", "smash"]
SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
os.makedirs(LOG_DIR, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = [   # name, model, use duration, mirror probability, frames shuffled in time (train + val)
    dict(name="A single pose (baseline)",        model="A", dur=False, mirror=0.5, shuf=False),
    dict(name="A  no mirror",                    model="A", dur=False, mirror=0.0, shuf=False),
    dict(name="B sequence (baseline)",           model="B", dur=True,  mirror=0.5, shuf=False),
    dict(name="B  no duration",                  model="B", dur=False, mirror=0.5, shuf=False),
    dict(name="B  no mirror",                    model="B", dur=True,  mirror=0.0, shuf=False),
    dict(name="B  frames shuffled in time",      model="B", dur=True,  mirror=0.5, shuf=True),
]

# ---------------- data (train + val only) ----------------
assert sorted(d for d in os.listdir(CLEAN_ROOT) if os.path.isdir(f"{CLEAN_ROOT}/{d}")) == CLASS_NAMES
man = pd.read_csv(MANIFEST)
sc = next(c for c in man.columns if "subject" in c.lower()); pc = next(c for c in man.columns if "split" in c.lower())
ns = lambda s: str(s).strip().lower() if str(s).strip().lower().startswith("p") else f"p{str(s).strip().lower()}"
subj2split = man.assign(_s=man[sc].map(ns)).drop_duplicates("_s").set_index("_s")[pc].to_dict()
recs = []
for ci, cname in enumerate(CLASS_NAMES):
    for p in sorted(glob.glob(f"{CLEAN_ROOT}/{cname}/*.npz")):
        d = np.load(p); stem = os.path.splitext(os.path.basename(p))[0]; subj = stem.split("_")[0].lower()
        recs.append(dict(seq=d["seq"], mid=d["mid_pose"], dur=float(d["duration"]), y=ci, subj=subj, split=subj2split[subj]))
df = pd.DataFrame(recs)
sets = {s: set(df[df.split == s].subj) for s in df.split.unique()}
assert len(df) == 1980 and sorted(df.groupby("split").size().tolist()) == [288, 288, 1404]
nm = list(sets); assert all(not (sets[a] & sets[b]) for i, a in enumerate(nm) for b in nm[i + 1:])
print("data + split check passed (no subject in two splits); test split not used\n")

def tens(split):
    s = df[df.split == split]
    return (torch.tensor(np.stack(s.seq), dtype=torch.float32), torch.tensor(np.stack(s.mid), dtype=torch.float32),
            torch.tensor(s.dur.values, dtype=torch.float32), torch.tensor(s.y.values, dtype=torch.long))
D = {s: tens(s) for s in ("train", "val")}
mu, sd_ = D["train"][2].mean(), D["train"][2].std()                 # duration statistics from TRAIN only
for s in D: D[s] = (D[s][0], D[s][1], (D[s][2] - mu) / sd_, D[s][3])

# ---------------- models + helpers ----------------
class PoseMLP(nn.Module):
    def __init__(self, n=12):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(34, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x, dur=None): return self.net(x)

class SeqCNN(nn.Module):
    def __init__(self, n=12, use_dur=True):
        super().__init__()
        self.use_dur = use_dur
        self.conv = nn.Sequential(
            nn.Conv1d(34, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256 + int(use_dur), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x, dur=None):
        h = self.conv(x.flatten(2).transpose(1, 2))
        h = torch.cat([h.mean(2), h.amax(2)], 1)
        if self.use_dur: h = torch.cat([h, dur.unsqueeze(1)], 1)
        return self.head(h)

def augment(x, mirror_p):
    B = x.shape[0]
    if mirror_p > 0:
        flip = torch.rand(B, device=x.device) < mirror_p
        if flip.any():
            xf = x[flip].clone(); xf[..., 0] = -xf[..., 0]; x[flip] = xf[..., SWAP, :]
    s = 1 + (torch.rand(B, device=x.device) * 2 - 1) * SCALE_JIT
    x = x * s.view(B, *([1] * (x.dim() - 1)))
    return x + torch.randn_like(x) * NOISE_STD

def tshuffle(x, gen=None):                       # scramble the order of the 64 time steps, per video
    B, T = x.shape[:2]
    idx = torch.rand(B, T, generator=gen).argsort(1).to(x.device)
    return torch.gather(x, 1, idx[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3]))

@torch.no_grad()
def evaluate(model, X, d, y):
    model.eval(); lg = model(X, d)
    return F.cross_entropy(lg, y).item(), (lg.argmax(1) == y).float().mean().item(), lg.argmax(1)

def acc_shuffled(model, X, d, y, reps=5):        # accuracy when a trained model sees scrambled frame order
    return float(np.mean([evaluate(model, tshuffle(X, torch.Generator().manual_seed(1000 + r)), d, y)[1] for r in range(reps)]))

def run(cfg, seed, keep_preds=False):
    torch.manual_seed(seed); np.random.seed(seed); g = torch.Generator().manual_seed(seed)
    isB = cfg["model"] == "B"; which = 0 if isB else 1
    model = (SeqCNN(use_dur=cfg["dur"]) if isB else PoseMLP()).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    Xtr, dtr, ytr = D["train"][which].to(dev), D["train"][2].to(dev), D["train"][3].to(dev)
    Xva, dva, yva = D["val"][which].to(dev), D["val"][2].to(dev), D["val"][3].to(dev)
    Xev = tshuffle(Xva, torch.Generator().manual_seed(999)) if cfg["shuf"] else Xva
    best, best_state, bad, best_ep, tr_acc = (-1.0, 9e9), None, 0, 0, 0.0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr), generator=g).to(dev); tc = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(Xtr[idx].clone(), cfg["mirror"])
            if cfg["shuf"]: xb = tshuffle(xb)
            lg = model(xb, dtr[idx]); loss = F.cross_entropy(lg, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tc += (lg.argmax(1) == ytr[idx]).sum().item()
        vl, va, _ = evaluate(model, Xev, dva, yva)
        if (va, -vl) > (best[0], -best[1]):
            best, best_state, bad, best_ep, tr_acc = (va, vl), copy.deepcopy(model.state_dict()), 0, ep, tc / len(Xtr)
        else:
            bad += 1
        if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    _, va, pr = evaluate(model, Xev, dva, yva)
    out = dict(config=cfg["name"], seed=seed, best_val=va, best_epoch=best_ep, train_acc=tr_acc,
               val_shuffled=acc_shuffled(model, Xva, dva, yva) if isB else np.nan)
    if keep_preds: out["preds"] = pr.cpu().numpy()
    return out

# ---------------- run everything ----------------
t0 = time.time(); rows, preds = [], {}
for cfg in CONFIGS:
    for sd in SEEDS:
        r = run(cfg, sd, keep_preds=(sd == SEEDS[0]))
        if "preds" in r: preds[cfg["name"]] = r.pop("preds")
        rows.append(r)
        print(f"{cfg['name']:30s} seed {sd}: val acc {r['best_val']:.3f} (epoch {r['best_epoch']:3d}) | {(time.time() - t0) / 60:.1f} min elapsed")
res = pd.DataFrame(rows); res.to_csv(f"{LOG_DIR}/ablation_runs.csv", index=False)

print("\n" + "=" * 100)
print("VALIDATION RESULTS, mean +- std over 5 seeds (test set NOT used). Pixel baseline (ResNet50, 1 frame): 0.2639")
print(f"{'config':30s}{'val acc':>16s}{'train acc':>12s}{'gap':>8s}{'val acc, frames scrambled':>28s}{'median best epoch':>20s}")
for name, g_ in res.groupby("config", sort=False):
    sh = g_.val_shuffled.mean()
    print(f"{name:30s}{g_.best_val.mean():>9.3f} +- {g_.best_val.std():.3f}{g_.train_acc.mean():>12.3f}"
          f"{g_.train_acc.mean() - g_.best_val.mean():>8.3f}{('n/a' if np.isnan(sh) else f'{sh:.3f}'):>28s}{g_.best_epoch.median():>20.0f}")
print("=" * 100)

yv = D["val"][3].numpy(); pa, pb = preds[CONFIGS[0]["name"]], preds[CONFIGS[2]["name"]]
print("\nPer-class validation recall (seed 42 baselines)")
print(f"{'class':22s}{'A pose':>9s}{'B sequence':>12s}{'videos':>8s}")
for c, n in enumerate(CLASS_NAMES):
    print(f"{n:22s}{(pa[yv == c] == c).mean():>9.2f}{(pb[yv == c] == c).mean():>12.2f}{int((yv == c).sum()):>8d}")
cm = np.zeros((12, 12), int); np.add.at(cm, (yv, pb), 1)
pairs = sorted([(cm[i, j], i, j) for i in range(12) for j in range(12) if i != j], reverse=True)[:8]
print("\nMost common mistakes of Model B (true -> predicted : count)")
for n, i, j in pairs: print(f"  {CLASS_NAMES[i]} -> {CLASS_NAMES[j]} : {n}")
print(f"\nSaved: {LOG_DIR}/ablation_runs.csv")
