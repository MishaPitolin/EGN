# EGN convergence experiments — working journal

## Setup

- Graph: 8 nodes, ring topology
- Edge types: `is_a` (type 0, bidirectional coherence/support) and `causes` (type 1, bidirectional conflict)
- Feature dim: 16
- Energy: `E = conflict + coherence - support`
- Base step size α = 0.3 (decays on smooth decrease down to 0.05)

All experiments in `egn_convergence.py`, plots in `results/`.

---

## Finding 1 — Message passing diverges on conflict graph

| Graph | Update | Final E | Behavior |
|-------|--------|---------|----------|
| peaceful | MP | ~steady | stable |
| conflict | MP | diverges | E drifts |

Message passing has no gradient link to E — it iterates a fixed point that does not exist when `causes` edges are present.

---

## Finding 2 — Full-batch GD converges monotonically

`x -= α·∇E` with `w=(1,1,1)`:

- E drops monotonically to ≈ −25 in 50 steps
- α decays to 0.05, then stays flat
- Equivalent to `weighted-gd` with `w=(1,1,1)` — zero difference

---

## Finding 3 — Weighted-GD sign convention matters

Weighted gradient: `F = Σ w_i · g_i`, update: `x -= α·F`.

The **support** term enters with a **minus** sign in E (`E = conflict + coherence − support`).  
Incorrect implementation: applying the minus sign as a negative weight → gradient *ascent* on support.  
Correct: differentiate `−support(x)` directly, then apply positive `w_support`.

- `wc=1.0, wk=1.0, ws=1.0` → matches GD (E ≈ −25)
- `wc=0.1, wk=1.0, ws=1.0` → plateau at E ≈ +1.27 (conflict under-emphasised)
- `wc=1.0, wk=1.0, ws=0.1` → conflict dominates, cos ≈ −0.81

---

## Finding 4 — Dynamic weights converge on 5/5 seeds (no shake needed)

`w_conflict(t) = sigmoid(5 · (||g_conflict(t)|| − 0.15))`

| Seed | Monotonic | Oscillations | Final E | Final w_conflict |
|------|-----------|-------------|---------|-----------------|
| 1 | Yes | No | −30.6 | 0.40 |
| 7 | Yes | No | −30.6 | 0.38 |
| 13 | Yes | No | −30.7 | 0.40 |
| 42 | Yes | No | −30.6 | 0.38 |
| 99 | Yes | No | −30.5 | 0.44 |

- All 5/5 seeds converge monotonically, no oscillations
- Final w_conflict in [0.38, 0.44], well above the sigmoid floor (0.32)
- Gradient norm ||g_c|| decays to ≈ 0.05–0.10
- **Shake mechanism not required** — weight adapts naturally

### Shake post-mortem

Three detectors tested, all failed for the same reason: the shake perturbs the system, which in turn pollutes the detector's own metric.

| Detector | Shakes / 50 steps | Problem |
|----------|------------------|---------|
| rel_change(E) < 1%, 3 steps | 8 | fires when E crosses zero |
| \|\|F\| not shrunk ≥2% / 5 steps | 24 | decay rate < 2%/5 steps |
| log-regression slope < 0.001 / 15 steps | 4→2→11* | shake creates the plateau it detects |

*With 15-step cooldown: 2 shakes / 50 steps → but periodic 16-step limit cycle emerges on longer runs.

Conclusion: **constant low conflict weight cannot be rescued by shake** — the weight itself is the bottleneck.

---

## Caveats

- **Single graph**: all tests on one 8-node ring with `is_a`+`causes` edges
- **Single topology**: no variations in node count, edge density, or graph structure
- **Single energy definition**: `E = conflict + coherence − support` (one specific decomposition)
- **Fixed hyperparameters**: k=5, th=0.15 — may need retuning for different graphs
- **α scheduler still active**: dynamic weights co-exist with the old meta-regulator (α decays on smooth decrease); interaction not analysed separately

These results are **not yet general properties of the architecture** — they are observations on one test graph.

---

## Plots

| File | Content |
|------|---------|
| `results/egn_convergence.png` | E(t) and α(t) for all modes + weight sensitivity + shake comparison |
| `results/egn_gradient_norm.png` | \|\|F\|(t) log-scale and E(t) for low-conflict weighted-GD (100 steps) |
| `results/egn_5seeds_comparison.png` | E(t) for dynamic weights on 5 random seeds — overlay |
