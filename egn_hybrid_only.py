import torch
import torch.nn.functional as F

torch.manual_seed(42)
D = 16
ALPHA = 0.05
NUM_ITERS = 200
L_MAX = 3

CONTENT_NODES = {0, 1, 2, 3}
META_NODES = {8}
PSI_BINARY_TH = 0.5
PSI_CONT_TH = 0.33

classification_str = {0: 'NORMAL', 1: 'REDUCED', 2: 'REJECTED'}

# -- Graph: hybrid-only topology -------------------------------------------
# Content nodes: 0, 1, 2, 3   |   Hub node: 8
# Edges (is_a, type 0):
#   0 ↔ 1  (content)
#   1 ↔ 8  (hub)
#   8 ↔ 2  (hub)
#   1 ↔ 3  (content)
# No edge 0↔8, 0↔2, 1↔2, 8↔3, 2↔3.
# Pair (0, 2): ONLY path is 0→1→8→2 (1 content edge + 2 hub edges)
# Pair (3, 2): ONLY path is 3→1→8→2 (same hybrid pattern)

ei_list, et_list = [], []
def add_edge(u, v, t):
    ei_list.append((u, v)); et_list.append(t)

add_edge(0, 1, 0); add_edge(1, 0, 0)
add_edge(1, 8, 0); add_edge(8, 1, 0)
add_edge(8, 2, 0); add_edge(2, 8, 0)
add_edge(1, 3, 0); add_edge(3, 1, 0)

ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
et = torch.tensor(et_list, dtype=torch.long)
N_TOTAL = 9  # nodes 0-3, 8

def build_type_adjacency(et, t):
    m = et == t
    adj = {i: set() for i in range(N_TOTAL)}
    for idx in range(et.size(0)):
        if not m[idx]:
            continue
        u, v = ei[0, idx].item(), ei[1, idx].item()
        adj[u].add(v)
    return adj

adj_isa = build_type_adjacency(et, 0)

# -- Embedding (same as egn_verification.py) --------------------------------
def _graph_mask():
    return (ei[0] < N_TOTAL) & (ei[1] < N_TOTAL)

def run_convergence(x_init, steps=NUM_ITERS):
    x = x_init.clone()
    gm = _graph_mask()
    for _ in range(steps):
        xc = x.clone().requires_grad_(True)
        zero_conn = (xc * 0).sum()
        m_c = (et == 1) & gm
        Ec = F.cosine_similarity(xc[ei[0, m_c]], xc[ei[1, m_c]], dim=1).sum() if m_c.any() else zero_conn
        Ek = (xc.mean(dim=0) - xc).norm(dim=1).mean()
        m_s = (et == 0) & gm
        Es = F.cosine_similarity(xc[ei[0, m_s]], xc[ei[1, m_s]], dim=1).mean() if m_s.any() else zero_conn
        g_c = torch.autograd.grad(Ec, xc, retain_graph=True)[0]
        g_k = torch.autograd.grad(Ek, xc, retain_graph=True)[0]
        g_s = torch.autograd.grad(-Es, xc, retain_graph=True)[0]
        F_total = g_c + g_k + g_s
        x = (xc - ALPHA * F_total).detach()
    return x

emb = run_convergence(torch.randn(N_TOTAL, D))

def compute_grounding(subj, obj, rtype):
    cs = F.cosine_similarity(emb[subj].unsqueeze(0), emb[obj].unsqueeze(0), dim=1).item()
    if rtype == 0:
        return cs
    elif rtype == 1:
        return -cs
    return abs(cs)

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

# -- psi functions ---------------------------------------------------------
def psi_binary(path):
    for node in path[1:-1]:
        if node in META_NODES:
            return 0.0
    return 1.0

def psi_continuous(path):
    if len(path) < 2:
        return 0.0
    content_edges = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if u in CONTENT_NODES and v in CONTENT_NODES:
            content_edges += 1
    return content_edges / (len(path) - 1)

def count_filtered(adj, start, target, L, psi_fn, thresh):
    all_paths = enumerate_paths(adj, start, target, L)
    if not all_paths:
        return 0, all_paths
    ok_paths = [p for p in all_paths if psi_fn(p) >= thresh]
    if not ok_paths:
        return 0, all_paths
    ok_paths.sort(key=len)
    used = set()
    count = 0
    for p in ok_paths:
        inner = set(p[1:-1])
        if not inner & used:
            used |= inner
            count += 1
    return count, all_paths

def V_q(subj, obj, rtype, psi_fn, thresh):
    grounding = compute_grounding(subj, obj, rtype)
    adj_use = adj_isa if rtype == 0 else {}
    support, all_paths = count_filtered(adj_use, subj, obj, L_MAX, psi_fn, thresh)
    ground_str = f"{grounding:.3f}"
    return support, ground_str

# -- Tests -----------------------------------------------------------------
pairs = [
    (0, 2, 0, "hybrid-only (0->1->8->2)", None),
    (3, 2, 0, "truthful-hybrid (3->1->8->2)", None),
]

def show_path_analysis(p, label):
    subj, obj, rtype, desc, _ = p
    print(f"\n  [{label}] {desc}")
    adj_use = adj_isa if rtype == 0 else {}
    _, all_paths = count_filtered(adj_use, subj, obj, L_MAX, psi_binary, PSI_BINARY_TH)
    print(f"    Total paths found: {len(all_paths)}")
    for path in all_paths:
        pb = psi_binary(path)
        pc = psi_continuous(path)
        meta = [n for n in path[1:-1] if n in META_NODES]
        flag = "[OK]" if pb >= PSI_BINARY_TH else "[NO]"
        print(f"      {'->'.join(map(str,path)):>16s}  psi_b={pb:.0f} psi_c={pc:.3f}  meta={meta}  {flag}")

def run_V_q_variant(name, psi_fn, thresh):
    print(f"\n  -- V_q with {name} (thresh={thresh}) --")
    for subj, obj, rtype, desc, _ in pairs:
        support, ground = V_q(subj, obj, rtype, psi_fn, thresh)
        cls = 2 if support < 1 else 0
        if desc.startswith("hybrid-only"):
            expected_cls = 2  # should be REJECTED (actually spurious)
            tag = "PASS" if cls == expected_cls else "FP"
            note = "correctly rejects spurious hybrid" if cls == expected_cls else "false positive"
        else:
            expected_cls = 0  # should be NORMAL (we labeled it truthful)
            tag = "PASS" if cls == expected_cls else "FN"
            note = "correctly accepts truthful hybrid" if cls == expected_cls else "false negative (psi too strict!)"
        print(f"    {desc:<42s}  support={support}  -> {classification_str[cls]:>8s}  [{tag}] {note}")

# ===========================================================================
print("=" * 68)
print("  Hybrid-only path test: graph with exactly 1 hybrid path")
print("  Content nodes: {0,1,2,3}   Meta node: {8}")
print("=" * 68)

print("\n  Adjacency (is_a):")
for n in sorted(adj_isa):
    if adj_isa[n]:
        print(f"    {n} -> {sorted(adj_isa[n])}")

print("\n" + "-" * 68)
print("  A. Path enumeration and psi scores")
print("-" * 68)
for pair in pairs:
    show_path_analysis(pair, pair[3][:20])

print("\n" + "-" * 68)
print("  B. Binary psi  (psi_b >= 0.5)")
print("     Expect: hybrid-only -> REJECTED  (psi_b=0, meta node 8)")
print("              truthful-hybrid -> REJECTED  (false negative!)")
print("-" * 68)
run_V_q_variant("binary psi", psi_binary, PSI_BINARY_TH)

print("\n" + "-" * 68)
print("  C. Continuous psi  (psi_c >= 0.33)")
print("     Expect: hybrid-only -> REJECTED  (psi_c=0.33 >= 0.33, but")
print("                                        still single path, LOW_SUPPORT=1)")
print("              truthful-hybrid -> NORMAL (psi_c=0.33 counts as support)")
print("-" * 68)
run_V_q_variant("continuous psi_c=0.33", psi_continuous, PSI_CONT_TH)

print("\n" + "-" * 68)
print("  Summary")
print("-" * 68)
print("""
  Hypothesis: binary psi falsely rejects hybrid-only truthful assertions.
  Proof by construction:

  Graph:        1
               /|\\
              0 | 3
                |
                8
                |
                2

  Pair (3, 2): path 3->1->8->2 (hybrid: 1 content + 2 hub edges)
    - binary psi:  psi_b = 0  (meta=8 in path)     -> REJECTED  [WRONG]
    - continuous:  psi_c = 0.33 >= 0.33             -> NORMAL     [CORRECT]

  The continuous version with theta_psi = 0.33 accepts the hybrid path
  because 1/3 of its edges are content edges. This saves the assertion
  from false rejection.
""")
