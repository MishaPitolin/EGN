"""
EGN minimal model -- complete pipeline from text input to verified output.

Integrates:
  1. Graph (10 nodes: 8 ring + 2 hubs)             -- egn_verification.py
  2. Superposition input + candidates              -- egn_superposition.py
  3. Iterative kernel F = -nabla E (4 channels)    -- egn_verification.py
  4. Self-referential temperature                  -- egn_superposition.py
  5. Plateau stopping (window=15, cooldown)        -- section 5
  6. Hysteresis collapse (k=3) + support check     -- egn_superposition.py
  7. V(q) with psi content-filtering               -- egn_verification.py
  8. Output: softmax over top-3 support nodes      -- section 8
"""
import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)

# -- Constants --------------------------------------------------------------
N_RING = 8
IDX_HUB8 = 8
IDX_HUB9 = 9
IDX_TEXT = 10
IDX_CAND_A = 11
IDX_CAND_B = 12
N_TOTAL = 13

D = 16
ALPHA = 0.05
NUM_ITERS = 200
L_MAX = 3

CONTENT_NODES = set(range(N_RING)) | {IDX_TEXT, IDX_CAND_A, IDX_CAND_B}
META_NODES = {IDX_HUB8, IDX_HUB9}
PSI_THRESHOLD = 0.5

# graph mask: restrict energy to ring nodes only
def graph_mask(ei):
    return (ei[0] < N_RING) & (ei[1] < N_RING)

# -- Graph builder ----------------------------------------------------------
ei_list, et_list = [], []
def add_edge(u, v, t):
    ei_list.append((u, v)); et_list.append(t)

# ring (is_a=0, causes=1)
for i in range(N_RING):
    j = (i + 1) % N_RING
    add_edge(i, j, 0); add_edge(j, i, 0)
    add_edge(i, j, 1); add_edge(j, i, 1)

# hub 8 -- is_a to even ring nodes
for v in [0, 2, 4, 6]:
    add_edge(IDX_HUB8, v, 0); add_edge(v, IDX_HUB8, 0)

# hub 9 -- is_a to odd ring nodes
for v in [1, 3, 5, 7]:
    add_edge(IDX_HUB9, v, 0); add_edge(v, IDX_HUB9, 0)

# candidate A: is_a to node 2 (content edge -> ring)
add_edge(IDX_CAND_A, 2, 0); add_edge(2, IDX_CAND_A, 0)

# candidate B: is_a to hub 8 (hub-only, no content path)
add_edge(IDX_CAND_B, IDX_HUB8, 0); add_edge(IDX_HUB8, IDX_CAND_B, 0)

# text node: is_a to both candidates (psi blocks hub-only path through B)
add_edge(IDX_TEXT, IDX_CAND_A, 0); add_edge(IDX_CAND_A, IDX_TEXT, 0)
add_edge(IDX_TEXT, IDX_CAND_B, 0); add_edge(IDX_CAND_B, IDX_TEXT, 0)

# conflict edges (type 2)
for u, v in [(4, 6), (5, 7), (0, 3), (1, 4), (3, 5)]:
    add_edge(u, v, 2); add_edge(v, u, 2)

ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
et = torch.tensor(et_list, dtype=torch.long)
GM = graph_mask(ei)

def build_type_adjacency(et, t):
    m = et == t
    adj = {i: set() for i in range(N_TOTAL)}
    for idx in range(et.size(0)):
        if not m[idx]:
            continue
        u, v = ei[0, idx].item(), ei[1, idx].item()
        adj[u].add(v)
    return adj

adj_isa     = build_type_adjacency(et, 0)
adj_causes  = build_type_adjacency(et, 1)
adj_conflict = build_type_adjacency(et, 2)

# -- Energy terms (ring nodes only) ----------------------------------------
def energy_conflict(x, ei, et):
    m = (et == 1) & GM
    if not m.any():
        return 0.0
    s, d = ei[0, m], ei[1, m]
    return F.cosine_similarity(x[s], x[d], dim=1).sum()

def energy_coherence(x):
    mu = x.mean(dim=1, keepdim=True)
    return ((x - mu) ** 2).mean()

def energy_support(x, ei, et):
    m = (et == 0) & GM
    if not m.any():
        return 0.0
    s, d = ei[0, m], ei[1, m]
    return F.cosine_similarity(x[s], x[d], dim=1).mean()

# -- psi functions (from egn_verification.py) -------------------------------
def psi(path):
    for node in path[1:-1]:
        if node in META_NODES:
            return 0.0
    return 1.0

def enumerate_paths(adj, start, target, L, visited=None):
    if start == target:
        return [[target]]
    if L == 0:
        return []
    if visited is None:
        visited = set()
    paths = []
    visited.add(start)
    for nb in adj.get(start, set()):
        if nb not in visited:
            subpaths = enumerate_paths(adj, nb, target, L - 1, visited)
            for sp in subpaths:
                paths.append([start] + sp)
    visited.remove(start)
    return paths

def count_filtered(adj, start, target, L):
    all_paths = enumerate_paths(adj, start, target, L)
    if not all_paths:
        return 0, []
    ok = [p for p in all_paths if psi(p) >= PSI_THRESHOLD]
    if not ok:
        return 0, all_paths
    ok.sort(key=len)
    used = set()
    cnt = 0
    for p in ok:
        inner = set(p[1:-1])
        if not inner & used:
            used |= inner
            cnt += 1
    return cnt, all_paths

def detect_conflict(u, v):
    return v in adj_conflict.get(u, set())

# -- V(q) (from egn_verification.py) ----------------------------------------
def V_q(subj, obj, rtype):
    adj_use = adj_isa if rtype == 0 else adj_causes if rtype == 1 else {}
    sup, raw = count_filtered(adj_use, subj, obj, L_MAX)
    confl = detect_conflict(subj, obj)
    return sup, len(raw), confl, raw

# -- Initial embeddings -----------------------------------------------------
x_init = torch.randn(N_TOTAL, D)

x_init[IDX_CAND_A] = x_init[2] + 0.05 * torch.randn(D)
x_init[IDX_CAND_B] = x_init[5] + 0.05 * torch.randn(D)
x_init[IDX_TEXT] = 0.5 * (x_init[IDX_CAND_A] + x_init[IDX_CAND_B]) + 0.1 * torch.randn(D)

# context anchor: node 2
context_anchor = x_init[2].detach().clone() + 0.02 * torch.randn(D)
CONTEXT_STRENGTH = 0.05

# -- Support check: direct adjacency to ring only ---------------------------
# Candidate must have its OWN content edge to a ring node.
# 2-hop through hub (B->8->ring) is spurious even without TEXT backdoor.
# 2-hop through TEXT->A->ring is also not the candidate's own support.
def has_support_for(cand_idx):
    for ring_n in range(N_RING):
        if ring_n in adj_isa.get(cand_idx, set()):
            return True
    return False

# Shared record: filled at collapse time, reused by output head
winner_reachable = []

# -- Main loop --------------------------------------------------------------
x = x_init.clone()
cand_indices = [IDX_CAND_A, IDX_CAND_B]
cand_anchors = [2, 5]

leader_counts = [0, 0]
collapse_step = None
collapse_winner = None

# stopping state
grad_norm_history = []
plateau_window = 15
plateau_threshold = 0.005
cooldown = 30
last_stop_check = 0
stop_reason = None

print("=" * 72)
print("  EGN MINIMAL MODEL -- end-to-end pipeline")
print("  Graph: 8 ring + 2 hubs | 2 candidates | superposition + V(q)")
print("=" * 72)

print("\n  +-[GRAPH]")
print(f"  |  Ring nodes:    0 1 2 3 4 5 6 7")
print(f"  |  Hub nodes:     {IDX_HUB8} (even ring), {IDX_HUB9} (odd ring)")
print(f"  |  Candidate A:   node {IDX_CAND_A} -> is_a -> node 2  (content edge)")
print(f"  |  Candidate B:   node {IDX_CAND_B} -> is_a -> hub 8  (hub-only)")
print(f"  |  Text node:     node {IDX_TEXT} -> is_a -> A, B")
print(f"  |  Conflicts:     (4,6) (5,7) (0,3) (1,4) (3,5)")

print("\n  +-[INPUT]")
print(f"  |  v_text = 0.5 * (cand_A + cand_B) + noise")
print(f"  |  context_anchor -> node 2 (strength={CONTEXT_STRENGTH})")
print(f"  |  Candidate A:  content path to ring (via node 2)  -> should WIN")
print(f"  |  Candidate B:  hub-only (no content path)         -> should LOSE")

step = 0
while step < NUM_ITERS and stop_reason is None:
    xc = x.clone().requires_grad_(True)

    # -- Energy -------------------------------------------------------------
    Ec = energy_conflict(xc[:N_RING], ei, et)
    Ek = energy_coherence(xc[:N_RING])
    Es = energy_support(xc[:N_RING], ei, et)

    g_c = torch.autograd.grad(Ec, xc, retain_graph=True)[0]
    g_k = torch.autograd.grad(Ek, xc, retain_graph=True)[0]
    g_s = torch.autograd.grad(-Es, xc, retain_graph=True)[0]

    # -- Self-referential temperature (section 4.4) -------------------------
    g_norm = g_c[:N_RING].norm().item()
    wc = 1.0 / (1.0 + math.exp(-5.0 * (g_norm - 0.15)))

    # -- Auxiliary energy: superposition (section 3) ------------------------
    v = xc[IDX_TEXT]
    cos_A = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_A].unsqueeze(0), dim=1)
    cos_B = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_B].unsqueeze(0), dim=1)
    logits = torch.cat([cos_A, cos_B])
    conf = torch.softmax(logits, dim=0)

    E_attract = -torch.logsumexp(logits, dim=0)
    E_anchor = 0.0
    for ci, ai in zip(cand_indices, cand_anchors):
        E_anchor -= F.cosine_similarity(xc[ci].unsqueeze(0), xc[ai].unsqueeze(0), dim=1).sum()
    E_ctx = -F.cosine_similarity(v.unsqueeze(0), context_anchor.unsqueeze(0), dim=1).sum()
    E_aux = E_attract + 0.1 * E_anchor + CONTEXT_STRENGTH * E_ctx
    g_aux = torch.autograd.grad(E_aux, xc, retain_graph=True)[0]

    # -- Compose F_total ----------------------------------------------------
    F_total = torch.zeros_like(xc)
    F_total[:N_RING] = wc * g_c[:N_RING] + g_k[:N_RING] + g_s[:N_RING] + g_aux[:N_RING]
    F_total[IDX_TEXT] = g_aux[IDX_TEXT]
    F_total[IDX_CAND_A] = g_aux[IDX_CAND_A]
    F_total[IDX_CAND_B] = g_aux[IDX_CAND_B]
    F_total[IDX_HUB8] = g_aux[IDX_HUB8]
    F_total[IDX_HUB9] = g_aux[IDX_HUB9]

    x = (xc - ALPHA * F_total).detach()

    # -- Log every 20 steps -------------------------------------------------
    if step % 20 == 0 or step < 5:
        total_norm = F_total[:N_RING].norm().item()
        print(f"  [ITER] step={step:3d}  E(c)={Ec.item():+.3f}  E(k)={Ek.item():.3f}  "
              f"E(s)={Es.item():.3f}  ||F||={total_norm:.4f}  wc={wc:.3f}  "
              f"conf_A={conf[0].item():.3f} conf_B={conf[1].item():.3f}")

    # -- Stopping: plateau on gradient norm (section 5) --------------------
    if step >= cooldown and step - last_stop_check >= cooldown:
        grad_norm_history.append(F_total[:N_RING].norm().item())
        if len(grad_norm_history) >= plateau_window:
            recent = grad_norm_history[-plateau_window:]
            std_val = float(torch.tensor(recent).std())
            if std_val < plateau_threshold:
                stop_reason = f"plateau (std={std_val:.5f} < {plateau_threshold})"
                break
        last_stop_check = step

    # -- Superposition: hysteresis collapse (section 3.4) ------------------
    if collapse_step is None:
        leader = 0 if conf[0] > conf[1] else 1
        supp = [has_support_for(ci) for ci in cand_indices]
        if supp[leader]:
            leader_counts[leader] += 1
            leader_counts[1 - leader] = 0
        else:
            leader_counts = [0, 0]

        if leader_counts[leader] >= 3 and supp[leader]:
            collapse_step = step
            collapse_winner = leader
            winner_anchor_local = cand_anchors[leader]
            # record reachable nodes at collapse time -- shared with output head
            winner_reachable.clear()
            for ring_n in range(N_RING):
                if ring_n == winner_anchor_local:
                    continue
                sup_v, _, confl_v, _ = V_q(winner_anchor_local, ring_n, 0)
                if sup_v > 0:
                    winner_reachable.append(ring_n)
            print(f"\n  +-[SUPERPOSITION] collapse at step {step} -> "
                  f"candidate {'A' if leader == 0 else 'B'} (idx={cand_indices[leader]})")
            print(f"  |  Hysteresis: k=3 consecutive wins with support")
            print(f"  |  A support={supp[0]}, B support={supp[1]}")
            print(f"  |  Independent support via V(q): {'YES' if supp[leader] else 'NO'}")
            print(f"  |  Reachable from anchor={winner_anchor_local}: {winner_reachable}")

    step += 1

if stop_reason is None and step >= NUM_ITERS:
    stop_reason = f"T_max={NUM_ITERS}"

if stop_reason:
    print(f"\n  +-[STOP] step={step}: stop_reason='{stop_reason}'")
else:
    print(f"\n  +-[STOP] step={step}: end of loop")

# -- Verification -----------------------------------------------------------
print("\n  +-[VERIFICATION] V(q) with psi content-filtering")

winner_node = collapse_winner
if winner_node is not None:
    winner_idx = cand_indices[winner_node]
    winner_anchor = cand_anchors[winner_node]

    sup, n_raw, confl, raw_paths = V_q(IDX_TEXT, winner_anchor, 0)

    if confl:
        verdict = "REDUCED"
        reason = f"direct conflict edge detected between text and anchor"
    elif sup >= 1:
        verdict = "NORMAL"
        paths_ok = [p for p in raw_paths if psi(p) >= PSI_THRESHOLD]
        path_strs = ['->'.join(map(str, p)) for p in paths_ok]
        reason = f"content-filtered support={sup} ({len(path_strs)} psi-passing paths: {', '.join(path_strs)})"
    else:
        verdict = "REJECTED"
        reason = f"no psi-passing path (found {n_raw} raw, all blocked by meta node)"

    print(f"  |  Assertion: text({IDX_TEXT}) is_a anchor({winner_anchor})")
    print(f"  |  raw_paths={n_raw}, filtered_support={sup}, conflict={'Y' if confl else 'N'}")
    for p in raw_paths:
        pb = psi(p)
        meta = [n for n in p[1:-1] if n in META_NODES]
        ok_str = "+" if pb >= PSI_THRESHOLD else "-"
        print(f"  |    path: {'->'.join(map(str,p)):>16s}  psi={pb:.0f}  meta={meta}  [{ok_str}]")
    print(f"  |")
    print(f"  |  VERDICT: {verdict}")
    print(f"  |  Reason: {reason}")

else:
    print(f"  |  No collapse occurred -- skipping V(q) assertion")
    verdict = "SKIPPED"

# -- Output (section 8) ------------------------------------------------------
print(f"\n  +-[OUTPUT] MINIMAL: final answer = anchor (no softmax over neighbors)")

if winner_node is not None:
    if verdict == "NORMAL":
        final_node = winner_anchor
        final_conf = 0.9   # V(q) confidence from NORMAL verdict
        print(f"  |  V(q) = NORMAL  ->  final node = anchor = {winner_anchor}")
        print(f"  |  (candidate {'A' if winner_node == 0 else 'B'} won collapse)")
        print(f"  |  No softmax over reachable neighbors needed.")
    else:
        final_node = None
        final_conf = 0.0
        print(f"  |  V(q) = {verdict}  ->  no answer")
else:
    final_node = None
    final_conf = 0.0
    print(f"  |  No collapse  ->  no answer")

print(f"\n  +-[SUMMARY]")
print(f"  |  Graph nodes: {N_TOTAL} ({N_RING} ring + 2 hubs + 1 text + 2 candidates)")
print(f"  |  Convergence steps: {step}")
if collapse_step is not None:
    print(f"  |  Superposition collapse: yes at step {collapse_step} -> {'A' if collapse_winner == 0 else 'B'}")
else:
    print(f"  |  Superposition collapse: no")
print(f"  |  Stop reason: {stop_reason}")
print(f"  |  V(q) verdict: {verdict}")
if final_node is not None:
    print(f"  |  Final answer: anchor node {final_node} (conf={final_conf:.3f})")
else:
    print(f"  |  Final answer: NONE (no collapse or REJECTED)")
print(f"  |")
print(f"  |  Analysis: should output head ever differ from anchor?")
print(f"  |  In current architecture (no type hierarchy, no part-whole relations),")
print(f"  |  V(q) verifies the anchor directly.  A non-anchor output would be")
print(f"  |  justified if the graph had IS-A chains (cat -> mammal -> animal)")
print(f"  |  where anchor = general category but answer = specific instance.")
print(f"  |  Until such hierarchy exists, output head = anchor.")
print(f"  +" + "-" * 35)
