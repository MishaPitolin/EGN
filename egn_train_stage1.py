"""
=============================================================================
  egn_train_stage1.py  --  EGN training, Stage 1: TransE on Wikidata5M
=============================================================================

  INSTRUCTIONS (Google Colab):
    1. Open https://colab.research.google.com/
    2. File -> Open notebook -> GitHub -> [your repo] -> egn_train_stage1.py
    3. Runtime -> Change runtime type -> T4 GPU  (free tier)
    4. Run all cells.
    5. When prompted, mount Google Drive (click link, paste token).
       Checkpoints are saved to /content/drive/MyDrive/EGN_data/checkpoints/
       every 30 minutes.

  What this script does:
    - Loads Wikidata5M via PyKEEN (cached to Drive, no re-download across sessions).
    - Trains TransE embeddings (d=64) with margin ranking loss.
    - Saves checkpoints every 30 min to survive Colab session drops.
    - Evaluates Hits@10 on a 1000-sample subset of test after each epoch.
=============================================================================
"""

# ---------------------------------------------------------------------------
# Cell 1: mount Drive & set environment
# ---------------------------------------------------------------------------
# @title 1. Mount Google Drive & set PYKEEN_HOME

import sys, subprocess, pkg_resources, os, math, time, random, itertools

# Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# PYKEEN_HOME on Drive so the dataset survives session restarts
PYKEEN_HOME = "/content/drive/MyDrive/EGN_data/pykeen_cache"
CHECKPOINT_DIR = "/content/drive/MyDrive/EGN_data/checkpoints"
os.makedirs(PYKEEN_HOME, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.environ["PYKEEN_HOME"] = PYKEEN_HOME

print(f"PYKEEN_HOME set to {PYKEEN_HOME}")
print(f"Checkpoints will be saved to {CHECKPOINT_DIR}")

# ---------------------------------------------------------------------------
# Cell 2: install dependencies
# ---------------------------------------------------------------------------
# @title 2. Install PyKEEN & PyTorch

_REQUIRED = {"pykeen", "torch"}

_installed = {d.key for d in pkg_resources.working_set}
_missing = _REQUIRED - _installed
if _missing:
    print(f"Installing: {_missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(_missing)])
    print("Done.")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cell 3: load Wikidata5M
# ---------------------------------------------------------------------------
# @title 3. Load Wikidata5M (cached on Drive)

from pykeen.datasets import Wikidata5M

print("Loading Wikidata5M (first download will cache to Drive)...")
t0 = time.time()
dataset = Wikidata5M(cache_root=PYKEEN_HOME)
print(f"Done in {time.time() - t0:.1f}s")

train_triples = dataset.training.mapped_triples    # (N_train, 3) LongTensor
valid_triples = dataset.validation.mapped_triples  # (N_valid, 3)
test_triples  = dataset.testing.mapped_triples     # (N_test, 3)

num_entities  = dataset.num_entities
num_relations = dataset.num_relations

print(f"  Entities:     {num_entities:,}")
print(f"  Relations:    {num_relations:,}")
print(f"  Train:        {train_triples.shape[0]:,}")
print(f"  Valid:        {valid_triples.shape[0]:,}")
print(f"  Test:         {test_triples.shape[0]:,}")

parameters_estimate = num_entities * 64 + num_relations * 64
print(f"  Model params (64-dim): {parameters_estimate:,}  "
      f"({parameters_estimate * 4 / 1024**3:.1f} GB float32)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

# Move triples to device
train_triples = train_triples.to(DEVICE)
valid_triples = valid_triples.to(DEVICE)
test_triples  = test_triples.to(DEVICE)

# ---------------------------------------------------------------------------
# Cell 4: model definition
# ---------------------------------------------------------------------------
# @title 4. Define EGNEmbeddingModel (TransE)

class EGNEmbeddingModel(nn.Module):
    def __init__(self, num_entities, num_relations, dim=64):
        super().__init__()
        self.dim = dim
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, s, r, o):
        # TransE: score = -||e(s) + e(r) - e(o)||_2
        # Higher = more plausible
        es = self.entity_embeddings(s)
        er = self.relation_embeddings(r)
        eo = self.entity_embeddings(o)
        return -torch.norm(es + er - eo, dim=1)

    def forward(self, s, r, o):
        return self.score(s, r, o)

model = EGNEmbeddingModel(num_entities, num_relations, dim=64).to(DEVICE)
print(f"Model on {DEVICE}: {sum(p.numel() for p in model.parameters()):,} parameters")

# ---------------------------------------------------------------------------
# Cell 5: training setup
# ---------------------------------------------------------------------------
# @title 5. Training loop

BATCH_SIZE = 1024
LR = 1e-3
MARGIN = 1.0
NUM_EPOCHS = 3
LOG_INTERVAL = 500          # batches
CHECKPOINT_INTERVAL = 1800  # seconds = 30 min
EVAL_SAMPLES = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
margin_loss = nn.MarginRankingLoss(margin=MARGIN)

# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------
def corrupt_o(s, r, o):
    """Replace o with a random entity (different from original)."""
    batch_size = s.shape[0]
    neg_o = torch.randint(0, num_entities, (batch_size,), device=DEVICE)
    same = neg_o == o
    while same.any():
        neg_o[same] = torch.randint(0, num_entities, (same.sum().item(),), device=DEVICE)
        same = neg_o == o
    return neg_o

# ---------------------------------------------------------------------------
# Hits@10 evaluator (subset of test)
# ---------------------------------------------------------------------------
def hits_at_10(model, test_triples, n_samples=EVAL_SAMPLES):
    model.eval()
    idx = torch.randperm(test_triples.shape[0], device=DEVICE)[:n_samples]
    batch = test_triples[idx]  # (N, 3)
    s, r, o = batch[:, 0], batch[:, 1], batch[:, 2]
    hits = 0
    with torch.no_grad():
        for i in range(s.shape[0]):
            # Score the true (s, r, o)
            true_score = model.score(s[i:i+1], r[i:i+1], o[i:i+1])
            # Score with 10 random corruptions of o
            candidates = []
            while len(candidates) < 10:
                neg = torch.randint(0, num_entities, (1,), device=DEVICE)
                if neg.item() != o[i].item():
                    candidates.append(neg.item())
            neg_t = torch.tensor(candidates, device=DEVICE)
            s_rep = s[i].expand(10)
            r_rep = r[i].expand(10)
            neg_scores = model.score(s_rep, r_rep, neg_t)
            # Rank true among 10 + 1 candidates
            all_scores = torch.cat([true_score, neg_scores])
            rank = (all_scores > true_score).sum().item() + 1
            if rank <= 10:
                hits += 1
    return hits / n_samples

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(epoch, batch_idx, loss_val):
    path = os.path.join(CHECKPOINT_DIR, f"egn_epoch{epoch}_batch{batch_idx}.pt")
    torch.save({
        "epoch": epoch,
        "batch": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_val,
    }, path)
    # Keep only the latest checkpoint (remove old ones)
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith("egn_") and fname != f"egn_epoch{epoch}_batch{batch_idx}.pt":
            os.remove(os.path.join(CHECKPOINT_DIR, fname))
    return path

def load_latest_checkpoint():
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("egn_")]
    if not ckpts:
        return 0, 0
    latest = max(ckpts, key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)))
    data = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=DEVICE)
    model.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    print(f"  Resumed from checkpoint: {latest} (epoch {data['epoch']}, batch {data['batch']})")
    return data["epoch"], data["batch"]

# ---------------------------------------------------------------------------
# Resume or start fresh
# ---------------------------------------------------------------------------
start_epoch, start_batch = load_latest_checkpoint()
global_batch = start_batch

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\nStarting training (epochs {start_epoch}+ to {NUM_EPOCHS})...")
last_ckpt_time = time.time()

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    perm = torch.randperm(train_triples.shape[0], device=DEVICE)
    train_shuffled = train_triples[perm]

    num_batches = (train_shuffled.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
    epoch_loss = 0.0
    epoch_start = time.time()

    for batch_idx in range(num_batches):
        lo = batch_idx * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, train_shuffled.shape[0])
        batch = train_shuffled[lo:hi]
        s, r, o = batch[:, 0], batch[:, 1], batch[:, 2]

        optimizer.zero_grad()

        # Positive scores
        pos_scores = model.score(s, r, o)

        # Negative: corrupt o
        neg_o = corrupt_o(s, r, o)
        neg_scores = model.score(s, r, neg_o)

        # Margin ranking loss: pos_scores should be > neg_scores
        target = torch.ones_like(pos_scores)
        loss = margin_loss(pos_scores, neg_scores, target)

        loss.backward()
        optimizer.step()

        global_batch += 1
        epoch_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0:
            elapsed = time.time() - epoch_start
            print(f"  [epoch {epoch}] batch {batch_idx:>6d}/{num_batches}  "
                  f"loss={loss.item():.4f}  elapsed={elapsed:.0f}s")

        # Checkpoint by time
        if time.time() - last_ckpt_time >= CHECKPOINT_INTERVAL:
            path = save_checkpoint(epoch, batch_idx, loss.item())
            print(f"  --> Checkpoint saved: {path}")
            last_ckpt_time = time.time()

    # End of epoch
    avg_loss = epoch_loss / num_batches
    epoch_time = time.time() - epoch_start
    print(f"\n  === Epoch {epoch} done === avg_loss={avg_loss:.4f}  time={epoch_time:.0f}s")

    # Save epoch-end checkpoint
    save_checkpoint(epoch, num_batches - 1, avg_loss)
    last_ckpt_time = time.time()

    # Hits@10 on test subset
    print(f"  Evaluating Hits@10 on {EVAL_SAMPLES} test samples...")
    t_eval = time.time()
    h10 = hits_at_10(model, test_triples, n_samples=EVAL_SAMPLES)
    print(f"  Hits@10 = {h10:.3f}  (eval time={time.time() - t_eval:.0f}s)")
    print()

print("Training complete.")
