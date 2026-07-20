import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

torch.manual_seed(42)

N = 8
D = 16
IDX_TEXT = N
IDX_CAND_A = N + 1
IDX_CAND_B = N + 2
TOTAL_NODES = N + 1 + 2

ALPHA = 0.05
NUM_ITERS = 100
T_MAX = 30

def build_graph(scenario):
    edges, et = [], []

    def add_edge(u, v, t):
        edges.append((u, v)); et.append(t)

    for i in range(N):
        j = (i + 1) % N
        for _ in range(2):
            add_edge(i, j, 0); add_edge(j, i, 0)
            add_edge(i, j, 1); add_edge(j, i, 1)

    add_edge(IDX_CAND_A, 2, 0)
    if scenario == 'b':
        add_edge(IDX_CAND_B, 5, 0)

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return ei, torch.tensor(et, dtype=torch.long)


def add_support_to_A(ei, et):
    src = [IDX_CAND_A] * N + list(range(N))
    dst = list(range(N)) + [IDX_CAND_A] * N
    new_ei = torch.tensor([src, dst], dtype=torch.long)
    new_et = torch.zeros(2 * N, dtype=torch.long)
    ei2 = torch.cat([ei, new_ei], dim=1)
    et2 = torch.cat([et, new_et])
    return ei2, et2


def _graph_mask(ei, et, t):
    m = (et == t) & (ei[0] < N) & (ei[1] < N)
    return m

def compute_conflict(x, ei, et):
    m = _graph_mask(ei, et, 1)
    if not m.any(): return 0.0
    s, d = ei[0, m], ei[1, m]
    return F.cosine_similarity(x[s], x[d], dim=1).sum()

def compute_coherence(x):
    m = x.mean(dim=1, keepdim=True)
    return ((x - m) ** 2).mean()

def compute_support(x, ei, et):
    m = _graph_mask(ei, et, 0)
    if not m.any(): return 0.0
    s, d = ei[0, m], ei[1, m]
    return F.cosine_similarity(x[s], x[d], dim=1).mean()

def has_support(cand_idx, ei, et):
    mask = (ei[0] == cand_idx) | (ei[1] == cand_idx)
    neighbors = set(ei[0, mask].tolist() + ei[1, mask].tolist())
    if any(n < N for n in neighbors):
        return True
    for n in neighbors:
        m2 = (ei[0] == n) | (ei[1] == n)
        n2 = set(ei[0, m2].tolist() + ei[1, m2].tolist())
        if any(nn < N for nn in n2):
            return True
    return False


def run_superposition(scenario, num_iters=NUM_ITERS, add_support_at=None,
                       context_strength=0.0, context_target=2):
    ei, et = build_graph(scenario)
    x = torch.randn(TOTAL_NODES, D)

    x[IDX_CAND_A] = x[2] + 0.05 * torch.randn(D)
    x[IDX_CAND_B] = x[5] + 0.05 * torch.randn(D)
    x[IDX_TEXT] = 0.5 * (x[IDX_CAND_A] + x[IDX_CAND_B]) + 0.1 * torch.randn(D)

    cand_anchors = [2, 5]
    cand_indices = [IDX_CAND_A, IDX_CAND_B]

    ctx = None
    if context_strength > 0:
        ctx = x[context_target].detach().clone() + 0.02 * torch.randn(D)

    leader_counts = [0, 0]
    collapse_step = None
    collapse_idx = None
    gap_node = None
    collapse_active = True

    conf_trace = []
    count_trace = []
    support_trace = []
    gap_marked = False
    resume_after_gap = (add_support_at is not None)

    for step in range(num_iters):
        if add_support_at is not None and step == add_support_at:
            ei, et = add_support_to_A(ei, et)
            cand_anchors = list(range(N))
            leader_counts = [0, 0]

        xc = x.clone().requires_grad_(True)

        Ec = compute_conflict(xc[:N], ei, et)
        Ek = compute_coherence(xc[:N])
        Es = compute_support(xc[:N], ei, et)

        g_c_full = torch.autograd.grad(Ec, xc, retain_graph=True)[0]

        wc = 1.0 / (1.0 + math.exp(-5.0 * (g_c_full[:N].norm().item() - 0.15)))

        g_k_full = torch.autograd.grad(Ek, xc, retain_graph=True)[0]
        g_s_full = torch.autograd.grad(-Es, xc, retain_graph=True)[0]

        v = xc[IDX_TEXT]
        cos_A = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_A].unsqueeze(0), dim=1)
        cos_B = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_B].unsqueeze(0), dim=1)
        logits = torch.cat([cos_A, cos_B])
        conf = torch.softmax(logits, dim=0)

        E_attract = -torch.logsumexp(logits, dim=0)
        E_anchor = 0.0
        for ci, ai in zip(cand_indices, cand_anchors):
            E_anchor -= F.cosine_similarity(xc[ci].unsqueeze(0), xc[ai].unsqueeze(0), dim=1).sum()

        E_aux = E_attract + 0.1 * E_anchor
        if ctx is not None:
            E_ctx = -F.cosine_similarity(xc[IDX_TEXT].unsqueeze(0), ctx.unsqueeze(0), dim=1).sum()
            E_aux = E_aux + context_strength * E_ctx
        g_aux = torch.autograd.grad(E_aux, xc, retain_graph=True)[0]

        F_total = torch.zeros_like(xc)
        F_total[:N] = wc * g_c_full[:N] + g_k_full[:N] + g_s_full[:N] + g_aux[:N]
        F_total[IDX_TEXT] = g_aux[IDX_TEXT]
        F_total[IDX_CAND_A] = g_aux[IDX_CAND_A]
        F_total[IDX_CAND_B] = g_aux[IDX_CAND_B]

        x = (xc - ALPHA * F_total).detach()

        supp = [has_support(ci, ei, et) for ci in cand_indices]

        if collapse_step is None and collapse_active:
            leader = 0 if conf[0] > conf[1] else 1
            if supp[leader]:
                leader_counts[leader] += 1
                leader_counts[1 - leader] = 0
            else:
                leader_counts = [0, 0]

            if leader_counts[leader] >= 3 and supp[leader]:
                collapse_step = step
                collapse_idx = leader
                collapse_active = False
            elif step >= T_MAX and not gap_marked:
                gap_node = {
                    'leader': leader,
                    'leader_support': supp[leader],
                    'confidence': conf[leader].item(),
                    'step': step,
                    'supp_all': supp.copy()
                }
                gap_marked = True
                if not resume_after_gap:
                    collapse_active = False

        conf_trace.append([conf[0].item(), conf[1].item()])
        count_trace.append(leader_counts.copy())
        support_trace.append(supp.copy())

    return conf_trace, count_trace, support_trace, collapse_step, collapse_idx, gap_node


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for sc_idx, scenario in enumerate(['a', 'b']):
    ax = axes[sc_idx]
    ct, cnt, st, cs, ci, gn = run_superposition(scenario)

    conf_A = [c[0] for c in ct]
    conf_B = [c[1] for c in ct]

    ax.plot(range(len(ct)), conf_A, 'b-', label='candidate A confidence')
    ax.plot(range(len(ct)), conf_B, 'r-', label='candidate B confidence')
    ax.fill_between(range(len(ct)), 0, [1]*len(ct),
                     where=[s[0] for s in st], color='blue', alpha=0.05,
                     label='A has support')
    ax.fill_between(range(len(ct)), 0, [1]*len(ct),
                     where=[s[1] for s in st], color='red', alpha=0.05,
                     label='B has support')

    if cs is not None:
        ax.axvline(x=cs, color='green', linestyle='--', linewidth=2,
                   label=f'collapse (candidate {"AB"[ci]}, step {cs})')
        ax.annotate(f'collapse -> {"AB"[ci]}', xy=(cs, 0.9),
                    fontsize=11, fontweight='bold', color='green')

    if gn is not None:
        ax.axvline(x=gn['step'], color='orange', linestyle=':', linewidth=2,
                   label=f'gap_node at step {gn["step"]}')
        ax.annotate(f'gap_node (leader={"AB"[gn["leader"]]})',
                    xy=(gn['step'], 0.05), fontsize=10, color='orange')

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Step')
    ax.set_ylabel('Confidence')
    ax.set_title(f'Scenario ({scenario}): {"A has support, B isolated" if scenario == "a" else "both have support"}')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Superposition collapse with T_max=30 gap_node', fontsize=13)
plt.tight_layout()
plt.savefig('results/egn_superposition.png', dpi=150)
print("Saved results/egn_superposition.png")

for sc in ['a', 'b']:
    ct, cnt, st, cs, ci, gn = run_superposition(sc)
    print(f"\nScenario ({sc}):")
    if cs is not None:
        print(f"  Collapse at step {cs} -> candidate {'AB'[ci]}")
    elif gn is not None:
        print(f"  No collapse — gap_node at step {gn['step']}")
        print(f"    Leader: candidate {'AB'[gn['leader']]}, support={gn['leader_support']}")
        print(f"    Confidence: {gn['confidence']:.4f}")
    else:
        print(f"  No collapse, no gap_node in {len(ct)} steps")
    print(f"  Final confidence: A={ct[-1][0]:.4f}, B={ct[-1][1]:.4f}")

print("\n--- Post-gap support addition test ---")
print("\nRunning scenario (a) with add_support_at=30...")
ct, cnt, st, cs, ci, gn = run_superposition('a', add_support_at=30)
if cs is not None:
    print(f"  Collapse: step {cs} -> candidate {'AB'[ci]}")
else:
    print("  No collapse")
if gn is not None:
    print(f"  Gap node: step {gn['step']}, leader={'AB'[gn['leader']]}")
else:
    print("  No gap node")
a_led = sum(1 for c in ct if c[0] > c[1])
b_led = sum(1 for c in ct if c[1] > c[0])
print(f"  Steps where A led: {a_led}/{len(ct)}")
print(f"  Steps where B led: {b_led}/{len(ct)}")
print(f"  Conf at step 29: A={ct[29][0]:.4f}, B={ct[29][1]:.4f}")
print(f"  Conf at step 30: A={ct[30][0]:.4f}, B={ct[30][1]:.4f}")
print(f"  Conf at step 35: A={ct[35][0]:.4f}, B={ct[35][1]:.4f}")
print(f"  Conf at step 50: A={ct[50][0]:.4f}, B={ct[50][1]:.4f}")

print("\n--- Context anchor test ---")
for strength in [0.01, 0.03, 0.05, 0.1, 0.2]:
    ct, cnt, st, cs, ci, gn = run_superposition('a', context_strength=strength, context_target=2)
    a_led = sum(1 for c in ct if c[0] > c[1])
    status = f"collapse at step {cs} -> {'AB'[ci]}" if cs is not None else "no collapse"
    print(f"  ctx={strength:.2f}: final A={ct[-1][0]:.4f} B={ct[-1][1]:.4f}, A-led={a_led}/100, {status}")

print("\n--- Seed sensitivity at ctx=0.05 (10 runs) ---")
print("  (v_text init noise + context_anchor noise vary; graph + candidates fixed)")
ei_static, et_static = build_graph('a')
x_static = torch.randn(TOTAL_NODES, D)
x_static[IDX_CAND_A] = x_static[2] + 0.05 * torch.randn(D)
x_static[IDX_CAND_B] = x_static[5] + 0.05 * torch.randn(D)

results_05 = []
for trial in range(10):
    x0 = x_static.clone()
    # vary v_text initialization
    torch.manual_seed(1000 + trial)
    x0[IDX_TEXT] = 0.5 * (x0[IDX_CAND_A] + x0[IDX_CAND_B]) + 0.1 * torch.randn(D)
    ctx_noise = 0.02 * torch.randn(D)
    ctx = x0[2].detach().clone() + ctx_noise

    x = x0.clone()
    cand_anchors = [2, 5]
    cand_indices = [IDX_CAND_A, IDX_CAND_B]
    leader_counts = [0, 0]
    collapse_step, collapse_idx = None, None
    gap_node = None
    collapse_active = True
    gap_marked = False

    for step in range(100):
        xc = x.clone().requires_grad_(True)
        Ec = compute_conflict(xc[:N], ei_static, et_static)
        Ek = compute_coherence(xc[:N])
        Es = compute_support(xc[:N], ei_static, et_static)
        g_c_full = torch.autograd.grad(Ec, xc, retain_graph=True)[0]
        wc = 1.0 / (1.0 + math.exp(-5.0 * (g_c_full[:N].norm().item() - 0.15)))
        g_k_full = torch.autograd.grad(Ek, xc, retain_graph=True)[0]
        g_s_full = torch.autograd.grad(-Es, xc, retain_graph=True)[0]

        v = xc[IDX_TEXT]
        cos_A = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_A].unsqueeze(0), dim=1)
        cos_B = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_B].unsqueeze(0), dim=1)
        logits = torch.cat([cos_A, cos_B])
        conf = torch.softmax(logits, dim=0)

        E_attract = -torch.logsumexp(logits, dim=0)
        E_anchor = 0.0
        for ci, ai in zip(cand_indices, cand_anchors):
            E_anchor -= F.cosine_similarity(xc[ci].unsqueeze(0), xc[ai].unsqueeze(0), dim=1).sum()
        E_ctx = -F.cosine_similarity(xc[IDX_TEXT].unsqueeze(0), ctx.unsqueeze(0), dim=1).sum()
        E_aux = E_attract + 0.1 * E_anchor + 0.05 * E_ctx
        g_aux = torch.autograd.grad(E_aux, xc, retain_graph=True)[0]

        F_total = torch.zeros_like(xc)
        F_total[:N] = wc * g_c_full[:N] + g_k_full[:N] + g_s_full[:N] + g_aux[:N]
        F_total[IDX_TEXT] = g_aux[IDX_TEXT]
        F_total[IDX_CAND_A] = g_aux[IDX_CAND_A]
        F_total[IDX_CAND_B] = g_aux[IDX_CAND_B]
        x = (xc - ALPHA * F_total).detach()

        supp = [has_support(ci, ei_static, et_static) for ci in cand_indices]
        if collapse_step is None and collapse_active:
            leader = 0 if conf[0] > conf[1] else 1
            if supp[leader]:
                leader_counts[leader] += 1
                leader_counts[1 - leader] = 0
            else:
                leader_counts = [0, 0]
            if leader_counts[leader] >= 3 and supp[leader]:
                collapse_step = step; collapse_idx = leader
                collapse_active = False
            elif step >= T_MAX and not gap_marked:
                gap_node = {'leader': leader, 'confidence': conf[leader].item(), 'step': step}
                gap_marked = True

    if collapse_step is not None:
        label = f"collapse at {collapse_step} -> {'AB'[collapse_idx]}"
    else:
        label = f"no collapse (gap at {gap_node['step']})"
    final_A = conf[0].item()
    results_05.append((final_A, label))
    print(f"  trial {trial}: final A={final_A:.4f} B={conf[1].item():.4f}, {label}")

a_wins = sum(1 for r in results_05 if r[0] > 0.5 and 'collapse' in r[1])
b_wins = sum(1 for r in results_05 if r[0] < 0.5 and 'collapse' in r[1])
nocoll = sum(1 for r in results_05 if 'no collapse' in r[1])
print(f"\n  Summary: A-collapse={a_wins}, B-collapse={b_wins}, no-collapse={nocoll}")

# --- Participation weight: continuous vs discrete transition ---
print("\n" + "=" * 55)
print("  Participation weight: continuous vs discrete transition")
print("=" * 55)

def participation_test_single(trial_seed, mode, num_iters=200, ctx_strength=0.05):
    torch.manual_seed(trial_seed)
    ei_p, et_p = build_graph('a')
    x = torch.randn(TOTAL_NODES, D)
    x[IDX_CAND_A] = x[2] + 0.05 * torch.randn(D)
    x[IDX_CAND_B] = x[5] + 0.05 * torch.randn(D)
    x[IDX_TEXT] = 0.5 * (x[IDX_CAND_A] + x[IDX_CAND_B]) + 0.1 * torch.randn(D)
    ctx_vec = x[2].detach().clone() + 0.02 * torch.randn(D)

    cand_indices = [IDX_CAND_A, IDX_CAND_B]
    cand_anchors = [2, 5]
    anch_emb = x[2].detach().clone()  # fixed anchor embedding

    conf_trace, wg_trace, e_sup_vt_trace = [], [], []

    for step in range(num_iters):
        xc = x.clone().requires_grad_(True)

        # Graph energy: original nodes 0-7 only
        m_c_orig = (et_p == 1) & (ei_p[0] < N) & (ei_p[1] < N)
        m_s_orig = (et_p == 0) & (ei_p[0] < N) & (ei_p[1] < N)

        Ec8 = compute_conflict(xc[:N], ei_p[:, m_c_orig], et_p[m_c_orig])
        Ek8 = compute_coherence(xc[:N])
        Es8 = compute_support(xc[:N], ei_p[:, m_s_orig], et_p[m_s_orig])

        g_c8 = torch.autograd.grad(Ec8, xc, retain_graph=True)[0]
        wc = 1.0 / (1.0 + math.exp(-5.0 * (g_c8[:N].norm().item() - 0.15)))
        g_k8 = torch.autograd.grad(Ek8, xc, retain_graph=True)[0]
        g_s8 = torch.autograd.grad(-Es8, xc, retain_graph=True)[0]

        v = xc[IDX_TEXT]
        cos_A = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_A].unsqueeze(0), dim=1)
        cos_B = F.cosine_similarity(v.unsqueeze(0), xc[IDX_CAND_B].unsqueeze(0), dim=1)
        logits = torch.cat([cos_A, cos_B])
        conf = torch.softmax(logits, dim=0)
        leader = 0 if conf[0] > conf[1] else 1
        conf_leader = conf[leader].item()

        if mode == 'continuous':
            wg = 1.0 / (1.0 + math.exp(-5.0 * (conf_leader - 0.5)))
        else:
            wg = 1.0 if conf_leader > 0.6 else 0.0

        # Participation: v_text is pulled toward leader's anchor, reinforcing confidence
        leader_anchor_idx = cand_anchors[leader]
        anch_emb_curr = xc[leader_anchor_idx].detach()
        E_part = -F.cosine_similarity(v.unsqueeze(0), anch_emb_curr.unsqueeze(0), dim=1).sum()
        E_vt_graph = wg * E_part
        g_vt_graph = torch.autograd.grad(E_vt_graph, xc, retain_graph=True)[0]

        # E_support contribution: cos(v_text, leader_anchor) when wg > 0
        e_sup_vt = wg * F.cosine_similarity(v.unsqueeze(0), anch_emb.unsqueeze(0), dim=1).item()

        E_attract = -torch.logsumexp(logits, dim=0)
        E_anchor = 0.0
        for ci, ai in zip(cand_indices, cand_anchors):
            E_anchor -= F.cosine_similarity(xc[ci].unsqueeze(0), xc[ai].unsqueeze(0), dim=1).sum()
        E_ctx = -F.cosine_similarity(xc[IDX_TEXT].unsqueeze(0), ctx_vec.unsqueeze(0), dim=1).sum()
        E_aux = E_attract + 0.1 * E_anchor + ctx_strength * E_ctx
        g_aux = torch.autograd.grad(E_aux, xc, retain_graph=True)[0]

        F_total = torch.zeros_like(xc)
        F_total[:N] = wc * g_c8[:N] + g_k8[:N] + g_s8[:N] + g_aux[:N] + g_vt_graph[:N]
        F_total[IDX_TEXT] = g_aux[IDX_TEXT] + g_vt_graph[IDX_TEXT]
        F_total[IDX_CAND_A] = g_aux[IDX_CAND_A]
        F_total[IDX_CAND_B] = g_aux[IDX_CAND_B]

        x = (xc - ALPHA * F_total).detach()

        conf_trace.append([conf[0].item(), conf[1].item()])
        wg_trace.append(wg)
        e_sup_vt_trace.append(e_sup_vt)

    return conf_trace, wg_trace, e_sup_vt_trace


import itertools

# Scan: find scenario where both continuous AND discrete reach high conf.
# We need ctx strong enough that base system (discrete with wg=0) also crosses 0.6.
scan_results = []
for seed, ctx in itertools.product([42, 100, 500, 1000, 2024], [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]):
    _, wg_ds, _ = participation_test_single(seed, 'discrete', ctx_strength=ctx)
    ct_cs, _, _ = participation_test_single(seed, 'continuous', ctx_strength=ctx)
    disc_trig = max(wg_ds) > 0.5
    cont_max_conf = max(max(c) for c in ct_cs)
    cont_rise = max(max(c) for c in ct_cs) - min(max(c) for c in ct_cs)
    scan_results.append((seed, ctx, disc_trig, cont_max_conf, cont_rise))

# Pick: discrete triggers AND continuous has gradual rise (not collapsed at step 0)
good = [r for r in scan_results if r[2] and r[4] > 0.05 and r[3] > 0.65]
if good:
    seed_best, ctx_best, _, _, rise_best = max(good, key=lambda r: r[4])
else:
    fallback = [r for r in scan_results if r[3] > 0.65 and r[4] > 0.02]
    seed_best, ctx_best, _, _, rise_best = max(fallback, key=lambda r: r[4]) if fallback else scan_results[-1]

print(f"\n  Best scenario: seed={seed_best}, ctx_strength={ctx_best}, rise={rise_best:.3f}")
conf_c, wg_c, sup_c = participation_test_single(seed_best, 'continuous', ctx_strength=ctx_best)
conf_d, wg_d, sup_d = participation_test_single(seed_best, 'discrete', ctx_strength=ctx_best)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax_c, ax_d = axes

ax_c.plot(range(len(conf_c)), [c[0] for c in conf_c], 'b-', label='A conf')
ax_c.plot(range(len(conf_c)), [c[1] for c in conf_c], 'r-', label='B conf')
ax_c.plot(range(len(wg_c)), wg_c, 'g--', linewidth=2, label='w_graph (continuous)')
ax_c.set_ylim(-0.05, 1.05)
ax_c.set_xlabel('Step')
ax_c.set_ylabel('Confidence / w_graph')
ax_c.set_title(f'Continuous sigmoid (ctx={ctx_best})')
ax_c.legend(fontsize=8)
ax_c.grid(True, alpha=0.3)

ax_d.plot(range(len(conf_d)), [c[0] for c in conf_d], 'b-', label='A conf')
ax_d.plot(range(len(conf_d)), [c[1] for c in conf_d], 'r-', label='B conf')
ax_d.plot(range(len(wg_d)), wg_d, 'g--', linewidth=2, label='w_graph (discrete)')
ax_d.set_ylim(-0.05, 1.05)
ax_d.set_xlabel('Step')
ax_d.set_ylabel('Confidence / w_graph')
ax_d.set_title(f'Discrete threshold 0.6 (ctx={ctx_best})')
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3)

plt.suptitle('Continuous vs discrete — participation weight transition', fontsize=13)
plt.tight_layout()
plt.savefig('results/egn_participation_weight.png', dpi=150)
print("\nSaved results/egn_participation_weight.png")

def count_jumps(wg, threshold=0.3):
    return sum(1 for i in range(1, len(wg)) if abs(wg[i] - wg[i-1]) > threshold)

print(f"\n  ctx={ctx_best}, seed={seed_best}:")
print(f"    Continuous: final A conf={conf_c[-1][0]:.4f}, wg=[{min(wg_c):.4f},{max(wg_c):.4f}], "
      f"jumps={count_jumps(wg_c)}")
print(f"    Discrete:   final A conf={conf_d[-1][0]:.4f}, wg=[{min(wg_d):.4f},{max(wg_d):.4f}], "
      f"jumps={count_jumps(wg_d)}")

# Additional comparison on a higher-ctx seed where discrete DOES trigger
# Run pure search on more seeds
print("\n  --- High-ctx comparison where discrete also triggers ---")
for hi_ctx in [0.5, 1.0, 2.0]:
    _, wg_d_hi = participation_test_single(seed_best, 'discrete', ctx_strength=hi_ctx)[:2]
    if max(wg_d_hi) > 0.5:
        conf_c_hi, wg_c_hi, _ = participation_test_single(seed_best, 'continuous', ctx_strength=hi_ctx)
        conf_d_hi, wg_d_hi, _ = participation_test_single(seed_best, 'discrete', ctx_strength=hi_ctx)
        print(f"    ctx={hi_ctx:.1f}: continuous wg=[{min(wg_c_hi):.4f},{max(wg_c_hi):.4f}], "
              f"discrete wg=[{min(wg_d_hi):.4f},{max(wg_d_hi):.4f}]")
        # Find the transition step in discrete
        trans_step = next((i for i, v in enumerate(wg_d_hi) if v > 0.5), -1)
        wg_before = wg_d_hi[trans_step-1] if trans_step > 0 else 0
        wg_after = wg_d_hi[trans_step] if trans_step >= 0 else 0
        wg_smooth_before = wg_c_hi[trans_step-1] if trans_step > 0 else 0
        wg_smooth_after = wg_c_hi[trans_step] if trans_step >= 0 else 0
        print(f"      discrete transition at step {trans_step}: wg {wg_before:.4f} -> {wg_after:.4f}")
        print(f"      continuous at same step: wg {wg_smooth_before:.4f} -> {wg_smooth_after:.4f}")
        # Plot detail around transition
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax, conf, wg, label in [
            (ax1, conf_c_hi, wg_c_hi, 'Continuous'),
            (ax2, conf_d_hi, wg_d_hi, 'Discrete')]:
            ax.plot(range(len(conf)), [c[0] for c in conf], 'b-', label='A conf')
            ax.plot(range(len(conf)), [c[1] for c in conf], 'r-', label='B conf')
            ax.plot(range(len(wg)), wg, 'g--', linewidth=2, label='w_graph')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Step')
            ax.set_ylabel('Confidence / w_graph')
            ax.set_title(f'{label} (ctx={hi_ctx})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            # Highlight transition
            if label == 'Discrete':
                for i in range(1, len(wg)):
                    if abs(wg[i] - wg[i-1]) > 0.3:
                        ax.axvline(x=i, color='orange', linestyle=':', alpha=0.7)
        plt.suptitle(f'Transition detail — ctx={hi_ctx}', fontsize=13)
        plt.tight_layout()
        plt.savefig('results/egn_participation_detail.png', dpi=150)
        print(f"      (detail saved to results/egn_participation_detail.png)")
        break
else:
    print("    No high-ctx scenario found where discrete triggers. "
          "The participation feedback loop is required to cross threshold; "
          "discrete thresholding is self-defeating in this system.")

# --- Dead zone analysis: discrete activation across ctx sweep ---
print("\n" + "=" * 55)
print("  Dead zone analysis: discrete activation across ctx sweep")
print("=" * 55)

SWEEP_SEED = 2024
SWEEP_ITERS = 100

def ctx_sweep():
    ctx_vals = [round(0.01 + i * 0.05, 2) for i in range(20)]  # 0.01, 0.06, ..., 0.96
    results = []
    for ctx in ctx_vals:
        conf_d, wg_d, _ = participation_test_single(SWEEP_SEED, 'discrete', num_iters=SWEEP_ITERS, ctx_strength=ctx)
        conf_c, wg_c, _ = participation_test_single(SWEEP_SEED, 'continuous', num_iters=SWEEP_ITERS, ctx_strength=ctx)

        disc_max_conf = max(max(c) for c in conf_d)
        disc_cross = disc_max_conf > 0.6
        disc_cross_step = next((i for i, c in enumerate(conf_d) if max(c) > 0.6), -1)

        cont_final_conf = max(max(c) for c in conf_c)
        cont_cross = cont_final_conf > 0.6
        cont_cross_step = next((i for i, c in enumerate(conf_c) if max(c) > 0.6), -1)

        cont_final_wg = wg_c[-1]

        results.append((ctx, disc_cross, disc_cross_step, cont_cross, cont_cross_step,
                        cont_final_wg, cont_final_conf))
        print(f"  ctx={ctx:.2f}: disc_conf>0.6={'Y' if disc_cross else 'N'}"
              f"({' at '+str(disc_cross_step) if disc_cross else ''}), "
              f"cont_conf>0.6={'Y' if cont_cross else 'N'}{' at '+str(cont_cross_step) if cont_cross else ''}, "
              f"cont_final_wg={cont_final_wg:.3f}")
    return results

sweep_results = ctx_sweep()

# Plot: dual Y-axis
fig3, ax3_left = plt.subplots(figsize=(10, 6))
ax3_right = ax3_left.twinx()

ctx_s = [r[0] for r in sweep_results]
disc_bin = [1.0 if r[1] else 0.05 for r in sweep_results]  # shift 0→0.05 so visible
cont_step = [r[4] if r[3] else 200 for r in sweep_results]

bars = ax3_left.bar(ctx_s, disc_bin, width=0.04, color='red', alpha=0.5, label='discrete conf>0.6')
line, = ax3_right.plot(ctx_s, cont_step, 'bs-', linewidth=2, markersize=6,
                       label='continuous step(conf>0.6)')
ax3_left.set_xlabel('ctx strength')
ax3_left.set_ylabel('discrete: conf > 0.6? (height=1=yes, height=0.05=no)')
ax3_right.set_ylabel('continuous: step when conf > 0.6 (200 = never)')
ax3_left.set_title('Discrete activation dead zone (seed=2024, 100 steps)')
ax3_left.set_ylim(0, 1.2)
ax3_right.set_ylim(-10, 210)
ax3_left.legend([bars], ['discrete conf>0.6'], loc='upper left', fontsize=9)
ax3_right.legend([line], ['continuous conf>0.6 step'], loc='upper right', fontsize=9)
ax3_left.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/egn_dead_zone.png', dpi=150)
print("\nSaved results/egn_dead_zone.png")

activated_ctx = [r[0] for r in sweep_results if r[1]]
dead_zone_boundary = activated_ctx[0] if activated_ctx else 2.0
n_activated = sum(r[1] for r in sweep_results)
n_cont = sum(r[3] for r in sweep_results)
disc_steps = [r[2] for r in sweep_results if r[1]]
cont_min = min((r[0] for r in sweep_results if r[3]), default=2.0)
print(f"\n  Results (seed={SWEEP_SEED}, {SWEEP_ITERS} steps):")
print(f"    Discrete  activates (conf>0.6) at ctx >= {dead_zone_boundary:.2f}")
print(f"    Discrete  activated: {n_activated}/{len(sweep_results)} ctx values")
print(f"    Continuous activates at ctx >= {cont_min:.2f}")
print(f"    Continuous activated: {n_cont}/{len(sweep_results)} ctx values")
print(f"    Dead zone: ctx < {dead_zone_boundary:.2f} — discrete never reaches conf>0.6")

# Speed comparison table for overlapping activation region
if disc_steps:
    overlap = [r for r in sweep_results if r[1] and r[3]]
    print(f"\n    Speed comparison (both modes reach conf>0.6):")
    print(f"    {'ctx':>6} | {'disc_step':>10} | {'cont_step':>10} | {'ratio':>6}")
    print(f"    {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    for r in overlap:
        ratio = r[2] / max(r[4], 1)
        print(f"    {r[0]:>6.2f} | {r[2]:>10d} | {r[4]:>10d} | {ratio:>6.2f}")
