# =====================================================================
# PHASE 3 + 4 - LOAD CLEANED SKELETONS, TRAIN MODEL A AND MODEL B
# Paste into ONE Colab cell. Needs /content/skeleton_clean and /content/split_manifest.csv
# The TEST split is loaded only so we can check the split is correct - it is NEVER used here.
# =====================================================================
import os, glob, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

CLEAN_ROOT = "/content/skeleton_clean"
MANIFEST   = "/content/split_manifest.csv"
OUT_DIR    = "/content/models_skeleton"
LOG_DIR    = "/content/results_skeleton"
SEED = 42
BATCH, MAX_EPOCHS, PATIENCE, LR, WD = 64, 150, 20, 1e-3, 1e-4
MIRROR_P, NOISE_STD, SCALE_JIT = 0.5, 0.02, 0.10   # augmentation (training data only)
USE_DURATION = True                                # Model B also sees how long the swing took
CLASS_NAMES = ["backhand", "backhand2hands", "backhand_slice", "backhand_volley", "flat_service",
               "forehand_flat", "forehand_openstands", "forehand_slice", "forehand_volley",
               "kick_service", "slice_service", "smash"]
SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # left<->right joint order (COCO-17)
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Phase 3: data + split ----------------
assert sorted(d for d in os.listdir(CLEAN_ROOT) if os.path.isdir(f"{CLEAN_ROOT}/{d}")) == CLASS_NAMES, "class folders differ from project class list"
man = pd.read_csv(MANIFEST)
sc = next(c for c in man.columns if "subject" in c.lower()); pc = next(c for c in man.columns if "split" in c.lower())
ns = lambda s: str(s).strip().lower() if str(s).strip().lower().startswith("p") else f"p{str(s).strip().lower()}"
subj2split = man.assign(_s=man[sc].map(ns)).drop_duplicates("_s").set_index("_s")[pc].to_dict()

recs = []
for ci, cname in enumerate(CLASS_NAMES):
    for p in sorted(glob.glob(f"{CLEAN_ROOT}/{cname}/*.npz")):
        d = np.load(p); stem = os.path.splitext(os.path.basename(p))[0]; subj = stem.split("_")[0].lower()
        recs.append(dict(seq=d["seq"], mid=d["mid_pose"], dur=float(d["duration"]), y=ci, subj=subj, split=subj2split[subj], stem=stem))
df = pd.DataFrame(recs)
sets = {s: set(df[df.split == s].subj) for s in df.split.unique()}
print("videos per split  :", df.groupby("split").size().to_dict())
print("subjects per split:", {s: len(v) for s, v in sets.items()})
assert len(df) == 1980 and sorted(df.groupby("split").size().tolist()) == [288, 288, 1404]
names = list(sets)
assert all(not (sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]), "a subject appears in 2 splits!"
print("split check passed: no subject appears in more than one split\n")

def tens(split):
    s = df[df.split == split]
    return (torch.tensor(np.stack(s.seq), dtype=torch.float32), torch.tensor(np.stack(s.mid), dtype=torch.float32),
            torch.tensor(s.dur.values, dtype=torch.float32), torch.tensor(s.y.values, dtype=torch.long))
D = {s: tens(s) for s in ("train", "val")}          # 'test' deliberately not built
dur_mu, dur_sd = D["train"][2].mean(), D["train"][2].std()          # statistics from TRAIN only
for s in D:
    D[s] = (D[s][0], D[s][1], (D[s][2] - dur_mu) / dur_sd, D[s][3])

# ---------------- models ----------------
class PoseMLP(nn.Module):                                   # Model A: one pose
    def __init__(self, n=12):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(34, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x, dur=None): return self.net(x)

class SeqCNN(nn.Module):                                    # Model B: whole pose sequence
    def __init__(self, n=12, use_dur=True):
        super().__init__()
        self.use_dur = use_dur
        self.conv = nn.Sequential(
            nn.Conv1d(34, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256 + int(use_dur), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x, dur=None):
        h = self.conv(x.flatten(2).transpose(1, 2))         # (B,64,17,2) -> (B,34,64)
        h = torch.cat([h.mean(2), h.amax(2)], 1)
        if self.use_dur: h = torch.cat([h, dur.unsqueeze(1)], 1)
        return self.head(h)

def augment(x):                                             # training data only
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < MIRROR_P
    if flip.any():
        xf = x[flip].clone(); xf[..., 0] = -xf[..., 0]; x[flip] = xf[..., SWAP, :]
    sc_ = 1 + (torch.rand(B, device=x.device) * 2 - 1) * SCALE_JIT
    x = x * sc_.view(B, *([1] * (x.dim() - 1)))
    return x + torch.randn_like(x) * NOISE_STD

@torch.no_grad()
def evaluate(model, X, d, y):
    model.eval(); lg = model(X, d)
    return F.cross_entropy(lg, y).item(), (lg.argmax(1) == y).float().mean().item()

def train(name, make_model, which):                         # which: 0 = sequence, 1 = single pose
    torch.manual_seed(SEED); np.random.seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    model = make_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    Xtr, dtr, ytr = D["train"][which].to(dev), D["train"][2].to(dev), D["train"][3].to(dev)
    Xva, dva, yva = D["val"][which].to(dev), D["val"][2].to(dev), D["val"][3].to(dev)
    best, best_state, bad, hist = (-1, 9e9), None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr), generator=g).to(dev); tl = tc = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(Xtr[idx].clone()); lg = model(xb, dtr[idx]); loss = F.cross_entropy(lg, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(idx); tc += (lg.argmax(1) == ytr[idx]).sum().item()
        vl, va = evaluate(model, Xva, dva, yva)
        hist.append(dict(epoch=ep, train_loss=tl / len(Xtr), train_acc=tc / len(Xtr), val_loss=vl, val_acc=va))
        if (va, -vl) > (best[0], -best[1]):
            best, best_state, bad = (va, vl), copy.deepcopy(model.state_dict()), 0
            torch.save(best_state, f"{OUT_DIR}/{name}.pt")
        else:
            bad += 1
        if ep % 10 == 0: print(f"  {name} epoch {ep:3d} | train acc {tc/len(Xtr):.3f} | val loss {vl:.3f} acc {va:.3f}")
        if bad >= PATIENCE: print(f"  {name}: early stop at epoch {ep}"); break
    h = pd.DataFrame(hist); h.to_csv(f"{LOG_DIR}/{name}_log.csv", index=False)
    b = h.loc[h.val_acc.idxmax()]
    print(f"{name}: best val acc {b.val_acc:.4f} (epoch {int(b.epoch)}, train acc then {b.train_acc:.3f}, {sum(p.numel() for p in model.parameters()):,} parameters)\n")
    return b

print("Training Model A (single pose at 50% of the swing)")
bA = train("model_A_pose", lambda: PoseMLP(), 1)
print("Training Model B (full pose sequence)")
bB = train("model_B_sequence", lambda: SeqCNN(use_dur=USE_DURATION), 0)

print("=" * 60)
print("VALIDATION accuracy (8 unseen people)  - test set NOT used")
print(f"  Pixels, 1 frame (ResNet50)   : 0.2639   (earlier result)")
print(f"  Skeleton, single pose (A)    : {bA.val_acc:.4f}")
print(f"  Skeleton, whole sequence (B) : {bB.val_acc:.4f}")
print("=" * 60)
