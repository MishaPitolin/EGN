"""
=============================================================================
  egn_data_stage1.py  —  EGN data pipeline, Stage 1: CoDEx-M loading
=============================================================================

  INSTRUCTIONS (Google Colab):
    1. Open https://colab.research.google.com/
    2. File -> Open notebook -> GitHub -> [your repo] -> egn_data_stage1.py
    3. Runtime -> Change runtime type -> T4 GPU  (free tier)
    4. Run all cells.

  What this script does:
    - Loads CoDEx-M from HuggingFace datasets (streaming=True, no disk save).
    - Builds entity2id and relation2id by scanning triples.
    - Prints entity count, relation count, triple count per split.
    - Computes parameter count for nn.Embedding(num_entities, 384) +
      nn.Embedding(num_relations, 384).

  Requirements (auto-installed in Colab):
    pip install datasets
=============================================================================
"""

# ---------------------------------------------------------------------------
# Colab setup — run this cell first
# ---------------------------------------------------------------------------
# @title 1. Install & imports
# @markdown Run this cell to install dependencies and import modules.

import sys, subprocess, pkg_resources, math

_REQUIRED = {"datasets"}

_installed = {d.key for d in pkg_resources.working_set}
_missing = _REQUIRED - _installed
if _missing:
    print(f"Installing missing packages: {_missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(_missing)])
    print("Done.\n")

import datasets
import torch

# ---------------------------------------------------------------------------
# 2. Load CoDEx-M — streaming only, no disk storage
# ---------------------------------------------------------------------------
# @title 2. Load CoDEx-M (streaming)
# @markdown Downloads nothing to disk; iterates triples on the fly.

print("Loading CoDEx-M (streaming)...")

_CONFIG_CANDIDATES = ["codex_m", "codex-m", "medium", "codex_medium"]
_loaded = False
for cfg in _CONFIG_CANDIDATES:
    try:
        dataset = datasets.load_dataset("codex", cfg, streaming=True, split=None)
        splits_available = list(dataset.keys())
        print(f"  Configuration '{cfg}' OK. Splits: {splits_available}")
        _loaded = True
        break
    except Exception as e:
        print(f"  Config '{cfg}' failed: {e}")
        continue

if not _loaded:
    # Last resort: load without config (dataset default)
    dataset = datasets.load_dataset("codex", streaming=True)
    if isinstance(dataset, dict):
        splits_available = list(dataset.keys())
    else:
        splits_available = ["train"]
        dataset = {"train": dataset}
    print(f"  Loaded default config. Splits: {splits_available}")

# ---------------------------------------------------------------------------
# 3. Build entity2id / relation2id by streaming all splits
# ---------------------------------------------------------------------------
# @title 3. Build entity2id & relation2id
# @markdown Scans every triple exactly once.

entity2id = {}
relation2id = {}
triple_count = {}

for split_name in splits_available:
    split = dataset[split_name]  # IterableDataset — streamed, not loaded in RAM
    cnt = 0
    for row in split:
        if "subject" in row:
            s, r, o = row["subject"], row["relation"], row["object"]
        else:
            s, r, o = row["head"], row["relation"], row["tail"]
        if s not in entity2id:
            entity2id[s] = len(entity2id)
        if o not in entity2id:
            entity2id[o] = len(entity2id)
        if r not in relation2id:
            relation2id[r] = len(relation2id)
        cnt += 1
    triple_count[split_name] = cnt

# ---------------------------------------------------------------------------
# 4. Statistics
# ---------------------------------------------------------------------------
# @title 4. Statistics & parameter estimate
# @markdown Embedding dimension = 384 (default for ComplEx / TuckER on CoDEx-M).

print("\n" + "=" * 60)
print("  CoDEx-M statistics")
print("=" * 60)
print(f"  Entities:        {len(entity2id):>10,}")
print(f"  Relations:       {len(relation2id):>10,}")
print(f"  Triples (total): {sum(triple_count.values()):>10,}")
print()

for s in splits_available:
    print(f"    {s:>12s}:  {triple_count[s]:>10,}")

DIM = 384
emb_entity_params    = len(entity2id)   * DIM
emb_relation_params  = len(relation2id) * DIM
total_params         = emb_entity_params + emb_relation_params

print()
print(f"  Embedding dimension: {DIM}")
print(f"  nn.Embedding(entities, {DIM})  -> {emb_entity_params:>12,} parameters")
print(f"  nn.Embedding(relations, {DIM}) -> {emb_relation_params:>12,} parameters")
print(f"  Total embedding params        {total_params:>12,}")
print(f"  Memory (float32):             {total_params * 4 / 1024**2:>8.2f} MB")
print()

# Sanity: first 5 entities, first 5 relations
print("  First 5 entities:")
for i, (e, _) in enumerate(list(entity2id.items())[:5]):
    print(f"    {i}:  {e}")
print()
print("  First 5 relations:")
for i, (r, _) in enumerate(list(relation2id.items())[:5]):
    print(f"    {i}:  {r}")

print("\nDone.  CoDEx-M is ready for EGN training stage 2.")
