"""
=============================================================================
  egn_data_stage1.py  --  EGN data pipeline, Stage 1: CoDEx-M loading
=============================================================================

  INSTRUCTIONS (Google Colab):
    1. Open https://colab.research.google.com/
    2. File -> Open notebook -> GitHub -> [your repo] -> egn_data_stage1.py
    3. Runtime -> Change runtime type -> T4 GPU  (free tier)
    4. Run all cells.

  What this script does:
    - Loads CoDEx-M directly from GitHub raw TSV (tsafavi/codex).
    - Builds entity2id and relation2id by scanning triples.
    - Prints entity count, relation count, triple count per split.
    - Computes parameter count for nn.Embedding(num_entities, 384) +
      nn.Embedding(num_relations, 384).

  Requirements (auto-installed in Colab):
    pip install pandas
=============================================================================
"""

# ---------------------------------------------------------------------------
# Colab setup -- run this cell first
# ---------------------------------------------------------------------------
# @title 1. Install & imports
# @markdown Run this cell to install dependencies and import modules.

import sys, subprocess, pkg_resources, math

_REQUIRED = {"pandas"}

_installed = {d.key for d in pkg_resources.working_set}
_missing = _REQUIRED - _installed
if _missing:
    print(f"Installing missing packages: {_missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(_missing)])
    print("Done.\n")

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# 2. Load CoDEx-M -- TSV from GitHub raw
# ---------------------------------------------------------------------------
# @title 2. Load CoDEx-M (GitHub raw TSV into memory -- < 5 MB total)
# @markdown Downloads three TSV files from tsafavi/codex; no permanent disk storage.

BASE_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m"
splits_available = ["train", "valid", "test"]
dataset = {}

print("Loading CoDEx-M from GitHub raw...")
for split_name in splits_available:
    url = f"{BASE_URL}/{split_name}.txt"
    df = pd.read_csv(url, sep="\t", header=None, names=["subject", "relation", "object"])
    dataset[split_name] = df
    print(f"  Loaded {split_name}: {len(df):,} triples")
print("Done.")

# ---------------------------------------------------------------------------
# 3. Build entity2id / relation2id
# ---------------------------------------------------------------------------
# @title 3. Build entity2id & relation2id

entity2id = {}
relation2id = {}
triple_count = {}

for split_name in splits_available:
    df = dataset[split_name]
    for s, r, o in zip(df["subject"], df["relation"], df["object"]):
        if s not in entity2id:
            entity2id[s] = len(entity2id)
        if o not in entity2id:
            entity2id[o] = len(entity2id)
        if r not in relation2id:
            relation2id[r] = len(relation2id)
    triple_count[split_name] = len(df)

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
