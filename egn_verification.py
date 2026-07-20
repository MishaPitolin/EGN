import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)
N_RING = 8
N_TOTAL = 11          # 8 ring + 3 hub nodes (8, 9, 10)
D = 16
ALPHA = 0.05
NUM_ITERS = 200
L_MAX = 3

CONTENT_NODES = set(range(8))
META_NODES = {8, 9, 10}
PSI_THRESHOLD = 0.5

assertion_type_str = {0: 'is_a', 1: 'causes', 2: 'conflict'}
classification_str = {0: 'NORMAL', 1: 'REDUCED', 2: 'REJECTED'}

# -- Graph builder ----------------------------------------------------------
def build_graph():
    ei_list, et_list = [], []
    def add_edge(u, v, t):
        ei_list.append((u, v)); et_list.append(t)
    # ring
    for i in range(N_RING):
        j = (i + 1) % N_RING
        add_edge(i, j, 0); add_edge(j, i, 0)
        add_edge(i, j, 1); add_edge(j, i, 1)
    # hub node 8  (is_a to 0, 2, 4, 6)
    for v in [0, 2, 4, 6]:
        add_edge(8, v, 0); add_edge(v, 8, 0)
    # hub node 9  (is_a to 1, 3, 5, 7)
    for v in [1, 3, 5, 7]:
        add_edge(9, v, 0); add_edge(v, 9, 0)
    # hub node 10 (is_a to all ring nodes — universal hub for double-spurious)
    for v in range(N_RING):
        add_edge(10, v, 0); add_edge(v, 10, 0)
    # conflict edges (type 2) for conflicting assertions
    conflict_pairs = [(4, 6), (5, 7), (0, 3), (1, 4), (3, 5)]
    for u, v in conflict_pairs:
        add_edge(u, v, 2); add_edge(v, u, 2)
    ei = torch.tensor(ei_list, dtype=torch.long).t().contiguous()
    et = torch.tensor(et_list, dtype=torch.long)
    return ei, et

def build_type_adjacency(et, t):
    m = et == t
    adj = {i: set() for i in range(N_TOTAL)}
    for idx in range(et.size(0)):
        if not m[idx]:
            continue
        u, v = ei[0, idx].item(), ei[1, idx].item()
        adj[u].add(v)
    return adj

ei, et = build_graph()

adj_isa = build_type_adjacency(et, 0)
adj_causes = build_type_adjacency(et, 1)
adj_conflict = build_type_adjacency(et, 2)

# -- Embedding convergence --------------------------------------------------
def _graph_mask():
    return (ei[0] < N_RING) & (ei[1] < N_RING)

def run_convergence(x_init, steps=NUM_ITERS):
    x = x_init.clone()
    gm = _graph_mask()
    for _ in range(steps):
        xc = x.clone().requires_grad_(True)
        m_c = (et == 1) & gm
        Ec = F.cosine_similarity(xc[ei[0, m_c]], xc[ei[1, m_c]], dim=1).sum() if m_c.any() else torch.tensor(0.0)
        Ek = (xc.mean(dim=0) - xc).norm(dim=1).mean()
        m_s = (et == 0) & gm
        Es = F.cosine_similarity(xc[ei[0, m_s]], xc[ei[1, m_s]], dim=1).mean() if m_s.any() else torch.tensor(0.0)
        g_c = torch.autograd.grad(Ec, xc, retain_graph=True)[0]
        g_k = torch.autograd.grad(Ek, xc, retain_graph=True)[0]
        g_s = torch.autograd.grad(-Es, xc, retain_graph=True)[0]
        F_total = g_c + g_k + g_s
        x = (xc - ALPHA * F_total).detach()
    return x

emb = run_convergence(torch.randn(N_TOTAL, D))

# -- Feature functions ------------------------------------------------------
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

def psi(path):
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

def analyze_paths(subj, obj, adj, label):
    raw = enumerate_paths(adj, subj, obj, L_MAX)
    if not raw:
        return
    raw.sort(key=len)
    print(f"\n    [{label}] all paths ({len(raw)}):")
    for p in raw:
        pb = psi(p)
        pc = psi_continuous(p)
        meta_nodes = [n for n in p[1:-1] if n in META_NODES]
        marker = "[OK]" if pb >= PSI_THRESHOLD else "[NO]"
        pstr = '->'.join(map(str,p))
        print(f"      {pstr:>18s}  psi={pb:.0f} psi_c={pc:.2f}  "
              f"meta={meta_nodes}  {marker}")

def count_vertex_disjoint_paths(adj, start, target, L):
    all_paths = enumerate_paths(adj, start, target, L)
    if not all_paths:
        return 0
    content_paths = [p for p in all_paths if psi(p) >= PSI_THRESHOLD]
    if not content_paths:
        return 0
    content_paths.sort(key=len)
    used = set()
    count = 0
    for p in content_paths:
        inner = set(p[1:-1])
        if not inner & used:
            used |= inner
            count += 1
    return count

def detect_direct_conflict(subj, obj):
    return obj in adj_conflict.get(subj, set())

# -- V(q) decision rule ------------------------------------------------
LOW_SUPPORT = 1   # need >= 1 content-filtered vertex-disjoint path

def V_q(subj, obj, rtype, debug=False):
    grounding = compute_grounding(subj, obj, rtype)
    adj_use = adj_isa if rtype == 0 else adj_causes if rtype == 1 else {}
    raw_paths = enumerate_paths(adj_use, subj, obj, L_MAX)
    raw_support = count_vertex_disjoint_paths(adj_use, subj, obj, L_MAX)

    content_paths = [p for p in raw_paths if psi(p) >= PSI_THRESHOLD]
    content_paths.sort(key=len)
    used = set()
    content_support = 0
    for p in content_paths:
        inner = set(p[1:-1])
        if not inner & used:
            used |= inner
            content_support += 1

    has_conflict = detect_direct_conflict(subj, obj)

    if debug:
        print(f"      raw_paths={len(raw_paths)}, content_paths={len(content_paths)}, "
              f"raw_sup={raw_support}, content_sup={content_support}")

    if has_conflict:
        return 1, 0.5, grounding, content_support, has_conflict
    if content_support < LOW_SUPPORT:
        return 2, 0.1, grounding, content_support, has_conflict
    return 0, 0.9, grounding, content_support, has_conflict

# -- Test assertions ----------------------------------------------------
truthful = [
    (0, 2, 0, "0 is_a 2 (ring 0-1-2 + hub 0-8-2)"),
    (0, 6, 0, "0 is_a 6 (ring 0-7-6 + hub 0-8-6)"),
    (1, 3, 0, "1 is_a 3 (ring 1-2-3 + hub 1-9-3)"),
    (1, 7, 0, "1 is_a 7 (ring 1-0-7 + hub 1-9-7)"),
    (2, 4, 0, "2 is_a 4 (ring 2-3-4 + hub 2-8-4)"),
]

fabricated = [
    (0, 4, 1, "0 causes 4 (no causes path, dist 4)"),
    (1, 5, 1, "1 causes 5 (no causes path, dist 4)"),
    (2, 6, 1, "2 causes 6 (no causes path, dist 4)"),
    (3, 7, 1, "3 causes 7 (no causes path, dist 4)"),
    (4, 0, 1, "4 causes 0 (no causes path, dist 4)"),
]

conflicting = [
    (4, 6, 0, "4 is_a 6 (ring+hub + confl(4,6))"),
    (5, 7, 1, "5 causes 7 (direct + confl(5,7))"),
    (0, 3, 1, "0 causes 3 (ring path + confl(0,3))"),
    (1, 4, 0, "1 is_a 4 (ring path + confl(1,4))"),
    (3, 5, 0, "3 is_a 5 (ring+hub + confl(3,5))"),
]

spurious = [
    (0, 4, 0, "0 is_a 4 (hub-only 0-8-4, no ring path)"),
    (1, 5, 0, "1 is_a 5 (hub-only 1-9-5, no ring path)"),
    (2, 6, 0, "2 is_a 6 (hub-only 2-8-6, no ring path)"),
    (3, 7, 0, "3 is_a 7 (hub-only 3-9-7, no ring path)"),
    (4, 0, 0, "4 is_a 0 (hub-only 4-8-0, no ring path)"),
]

double_spurious = [
    (0, 4, 0, "ds: 0 is_a 4 (2 hub paths: 8+10)"),
    (1, 5, 0, "ds: 1 is_a 5 (2 hub paths: 9+10)"),
    (2, 6, 0, "ds: 2 is_a 6 (2 hub paths: 8+10)"),
    (3, 7, 0, "ds: 3 is_a 7 (2 hub paths: 9+10)"),
    (4, 0, 0, "ds: 4 is_a 0 (2 hub paths: 8+10)"),
]

# -- Hybrid-spurious: mixed paths (content edge + hub edge) -------------------
# Pairs with ring distance = 4 (no ring-only path within L=3).
# Paths like 0→1→10→4 have ONE content edge (0→1) + TWO hub edges (1→10, 10→4).
# Current ψ rejects ALL paths due to the meta node — even the mixed ones.
hybrid_spurious = [
    (0, 4, 0, "hy: 0-1-10-4  (1 content + 2 hub edges)"),
    (1, 5, 0, "hy: 1-2-10-5  (1 content + 2 hub edges)"),
    (2, 6, 0, "hy: 2-3-10-6  (1 content + 2 hub edges)"),
]

all_sets = [
    ("TRUTHFUL", truthful),
    ("FABRICATED", fabricated),
    ("CONFLICTING", conflicting),
    ("SPURIOUS", spurious),
    ("DOUBLE_SPURIOUS", double_spurious),
    ("HYBRID_SPURIOUS", hybrid_spurious),
]

print("=" * 70)
print("  V(q) with psi content-filtering  (8-node ring + 3 hubs, seed=42)")
print(f"  LOW_SUPPORT={LOW_SUPPORT}, PSI_THRESHOLD={PSI_THRESHOLD}")
print("=" * 70)

matrix = {}

for set_name, assertions in all_sets:
    print(f"\n  -- {set_name} ({len(assertions)} assertions) --")
    set_results = []
    for subj, obj, rtype, desc in assertions:
        cls, conf, grounding, support, has_conflict = V_q(subj, obj, rtype)
        rstr = f"{classification_str[cls]:>8} (conf={conf:.2f}, gr={grounding:.3f}, csup={support}, "
        rstr += f"cnfl={'Y' if has_conflict else 'N'})"
        set_results.append(cls)
        print(f"    {desc:<48s}  -> {rstr}")
    if set_name == "HYBRID_SPURIOUS":
        print(f"\n    psi analysis per path:")
        for subj, obj, rtype, desc in assertions:
            adj_use = adj_isa if rtype == 0 else adj_causes if rtype == 1 else {}
            analyze_paths(subj, obj, adj_use, desc[:20])
    matrix[set_name] = set_results

# -- Confusion-style summary ------------------------------------------------
print("\n" + "=" * 70)
print("  Confusion matrix (content-filtered support, LOW_SUPPORT >= 1)")
print("=" * 70)

header = f"{'':>14} | {'NORMAL':>8} {'REDUCED':>8} {'REJECTED':>8}"
sep = f"{'-'*14}-+-{'-'*8}-{'-'*8}-{'-'*8}"
print(f"\n  {header}")
print(f"  {sep}")

for set_name in ["TRUTHFUL", "FABRICATED", "CONFLICTING", "SPURIOUS", "DOUBLE_SPURIOUS", "HYBRID_SPURIOUS"]:
    res = matrix[set_name]
    n = len(res)
    norm = sum(1 for c in res if c == 0)
    red = sum(1 for c in res if c == 1)
    rej = sum(1 for c in res if c == 2)
    print(f"  {set_name:>16} | {norm:>3}/{n:<4}  {red:>3}/{n:<4}  {rej:>3}/{n:<4}")

print(f"\n  Expected: truthful -> NORMAL, fabricated -> REJECTED, conflicting -> REDUCED,")
print(f"            spurious -> REJECTED, double_spurious -> REJECTED,")
print(f"            hybrid_spurious -> REJECTED  (mixed paths also blocked by psi)")
print(f"  {'':>16}   {'-'*30}")
for sname, correct_idx in [("TRUTHFUL", 0), ("FABRICATED", 2), ("CONFLICTING", 1),
                            ("SPURIOUS", 2), ("DOUBLE_SPURIOUS", 2),
                            ("HYBRID_SPURIOUS", 2)]:
    total = len(matrix[sname])
    correct = sum(1 for c in matrix[sname] if c == correct_idx)
    result = "PASS" if correct == total else "FAIL"
    print(f"  {sname+'-'+classification_str[correct_idx]:>32} = {correct}/{total}  [{result}]")

print(f"\n  NOTE: psi(path) rejects paths with meta-nodes (hubs 8,9,10) as intermediate")
print(f"  vertices. Only paths through content nodes (ring 0-7) count toward support.")
print(f"  This splits SPURIOUS and DOUBLE_SPURIOUS (all paths go through hubs) from")
print(f"  TRUTHFUL (paths through ring nodes only).")
print(f"")
print(f"  psi_continuous(path) shows the fraction of content edges in each path.")
print(f"  Hybrid paths (0-1-10-4) score psi_c=0.33 but still fail psi_binary=0")
print(f"  because ANY meta node in the path blocks the whole path. This is correct")
print(f"  for hub paths since meta nodes carry no compositional semantics, but may")
print(f"  be too strict if content edges form a meaningful partial relation.")
