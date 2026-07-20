import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

N = 8
D = 16
NUM_ITERS = 50
ALPHA_INIT = 0.3
ALPHA_MIN = 0.05
ALPHA_MAX = 0.9
PLATEAU_STEPS = 3


def build_graph(mode):
    edges = []
    edge_types = []

    def add_edge(u, v, t):
        edges.append((u, v))
        edge_types.append(t)

    # ring: bidirectional is_a edges (type 0) between consecutive nodes
    for i in range(N):
        j = (i + 1) % N
        add_edge(i, j, 0)
        add_edge(j, i, 0)

    if mode == 'conflict':
        # same pairs also get causes edges (type 1)
        for i in range(N):
            j = (i + 1) % N
            add_edge(i, j, 1)
            add_edge(j, i, 1)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    return edge_index, edge_type


def typed_message_passing(x, edge_index, edge_type, type_weight):
    row, col = edge_index
    num_types = type_weight.shape[0]
    msg = torch.zeros_like(x)
    for t in range(num_types):
        mask = edge_type == t
        if not mask.any():
            continue
        src = row[mask]
        dst = col[mask]
        neigh = x[src]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, neigh)
        counts = torch.zeros(N, dtype=torch.float)
        counts.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        counts = counts.clamp(min=1)
        mean_msgs = out / counts.unsqueeze(1)
        msg = msg + type_weight[t] * mean_msgs
    return msg


def compute_conflict(x, edge_index, edge_type):
    causes_mask = edge_type == 1
    if causes_mask.any():
        src = edge_index[0, causes_mask]
        dst = edge_index[1, causes_mask]
        return torch.cosine_similarity(x[src], x[dst], dim=1).sum()
    return 0.0


def compute_coherence(x):
    xi_mean = x.mean(dim=1, keepdim=True)
    return ((x - xi_mean) ** 2).mean()


def compute_support(x, edge_index, edge_type):
    is_a_mask = edge_type == 0
    if is_a_mask.any():
        src = edge_index[0, is_a_mask]
        dst = edge_index[1, is_a_mask]
        return torch.cosine_similarity(x[src], x[dst], dim=1).mean()
    return 0.0


def compute_energy(x, edge_index, edge_type):
    return compute_conflict(x, edge_index, edge_type) + \
           compute_coherence(x) - \
           compute_support(x, edge_index, edge_type)


def detect_plateau(vals, n):
    if len(vals) < n + 1:
        return False
    return all(vals[-i - 1] >= vals[-i - 2] for i in range(n))


def detect_smooth_decrease(vals, n):
    if len(vals) < n + 1:
        return False
    return all(vals[-i - 1] < vals[-i - 2] for i in range(n))


def detect_plateau_v2(grad_norms, n, threshold):
    if len(grad_norms) < n + 1:
        return False
    ratio = grad_norms[-1] / max(grad_norms[-n - 1], 1e-12)
    return ratio > (1.0 - threshold)


def detect_plateau_trend(grad_norms, window, slope_threshold):
    if len(grad_norms) < window:
        return False
    recent = grad_norms[-window:]
    y = [max(v, 1e-12) for v in recent]
    import math
    y = [math.log(v) for v in y]
    n = window
    t = list(range(n))
    sum_t = n * (n - 1) / 2
    sum_y = sum(y)
    sum_tt = n * (n - 1) * (2 * n - 1) / 6
    sum_ty = sum(ti * yi for ti, yi in zip(t, y))
    slope = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t)
    return abs(slope) < slope_threshold


def run_simulation(graph_mode, update_mode='mp', x_init=None, weights=None, shake=False, num_iters=None,
                   dynamic_weights=False, dw_k=5.0, dw_threshold=0.15):
    edge_index, edge_type = build_graph(graph_mode)
    x = torch.randn(N, D) if x_init is None else x_init.clone()
    num_types = edge_type.unique().numel() if edge_type.numel() > 0 else 0
    type_weight = nn.Parameter(torch.ones(max(num_types, 1)))
    max_steps = num_iters if num_iters is not None else NUM_ITERS

    w = {'conflict': 1.0, 'coherence': 1.0, 'support': 1.0}
    if weights is not None:
        w.update(weights)
    w_base = w.copy()

    alpha = ALPHA_INIT
    energies = []
    alphas = []
    f_norms = []
    shake_next = False
    shake_just_applied = False
    shake_count = 0
    shake_steps = []
    shake_cooldown = 0
    label = f'{graph_mode}-{update_mode}'
    if weights is not None:
        label += f' wc={w["conflict"]:.1f} ws={w["support"]:.1f}'
    if dynamic_weights:
        label += f' dw(k={dw_k},th={dw_threshold})'
    if shake:
        label += ' shake'

    for step in range(max_steps):
        # decrement cooldown
        if shake_cooldown > 0:
            shake_cooldown -= 1

        # apply shake override if triggered
        if shake and shake_next:
            w_active = {'conflict': 1.0, 'coherence': 1.0, 'support': 1.0}
            shake_next = False
            shake_just_applied = True
            shake_count += 1
            shake_steps.append(step)
            shake_cooldown = 15
        else:
            w_active = w
            shake_just_applied = False

        if update_mode == 'gd':
            xc = x.clone().requires_grad_(True)
            E_loss = compute_energy(xc, edge_index, edge_type)
            E_loss.backward()
            f_norms.append(xc.grad.norm().item())
            x = (xc - alpha * xc.grad).detach()
        elif update_mode == 'weighted-gd':
            xc = x.clone().requires_grad_(True)
            E_c = compute_conflict(xc, edge_index, edge_type)
            E_coh = compute_coherence(xc)
            E_s = compute_support(xc, edge_index, edge_type)
            g_c = torch.autograd.grad(E_c, xc, retain_graph=True)[0]
            if dynamic_weights:
                import math
                g_c_norm = g_c.norm().item()
                w_active['conflict'] = 1.0 / (1.0 + math.exp(-dw_k * (g_c_norm - dw_threshold)))
            g_coh = torch.autograd.grad(E_coh, xc, retain_graph=True)[0]
            g_s = torch.autograd.grad(-E_s, xc, retain_graph=True)[0]
            F = w_active['conflict'] * g_c + w_active['coherence'] * g_coh + w_active['support'] * g_s
            f_norms.append(F.norm().item())
            x = (xc - alpha * F).detach()
        else:
            msg = typed_message_passing(x, edge_index, edge_type, type_weight)
            x = ((1 - alpha) * x + alpha * torch.tanh(msg)).detach()
            f_norms.append(0.0)  # not applicable

        E = compute_energy(x, edge_index, edge_type).item()
        energies.append(E)
        alphas.append(alpha)

        # stuck detection v3: slope of log||F|| over last 15 steps near zero
        if shake and shake_cooldown == 0 and len(f_norms) >= 15:
            if detect_plateau_trend(f_norms, 15, 0.001):
                shake_next = True

        if detect_plateau(energies, PLATEAU_STEPS):
            alpha = min(alpha * 1.2, ALPHA_MAX)
        elif detect_smooth_decrease(energies, PLATEAU_STEPS):
            alpha = max(alpha * 0.9, ALPHA_MIN)

        marker = '  <SHAKE>' if shake_just_applied else ''
        print(f"[{label:44s}] step {step:2d}  E = {E:.6f}  alpha = {alpha:.4f}{marker}")

    if shake:
        print(f"  => total shakes applied: {shake_count}, steps: {shake_steps}")
    return energies, alphas, x, edge_index, edge_type, shake_steps, f_norms


def report_cos_sim(x, edge_index, edge_type, tag):
    for t, name in [(0, 'is_a'), (1, 'causes')]:
        mask = edge_type == t
        if not mask.any():
            continue
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        cos = torch.cosine_similarity(x[src], x[dst], dim=1)
        print(f"  [{tag}] {name}: mean cos = {cos.mean().item():+.4f}  "
              f"min = {cos.min().item():+.4f}  max = {cos.max().item():+.4f}")


print("=== Peaceful graph, message passing ===\n")
E_peace_mp, a_peace_mp, *_ = run_simulation('peaceful', 'mp')

print("\n" + "=" * 50 + "\n")

print("=== Conflict graph, message passing ===\n")
E_conflict_mp, a_conflict_mp, *_ = run_simulation('conflict', 'mp')

print("\n" + "=" * 50 + "\n")

print("=== Conflict graph, gradient descent ===\n")
E_conflict_gd, a_conflict_gd, *_ = run_simulation('conflict', 'gd')

# --- Weight sensitivity: three weighted-GD runs with same x0 ---
x0 = torch.randn(N, D)

weight_sets = [
    ('conflict (1,1,1)', {'conflict': 1.0, 'coherence': 1.0, 'support': 1.0}),
    ('low-conflict  (0.1,1,1)', {'conflict': 0.1, 'coherence': 1.0, 'support': 1.0}),
    ('low-support   (1,1,0.1)', {'conflict': 1.0, 'coherence': 1.0, 'support': 0.1}),
]

results = {}
for tag, w in weight_sets:
    print("\n" + "=" * 50)
    print(f"=== Conflict graph, weighted-GD {tag} ===\n")
    res = run_simulation('conflict', 'weighted-gd', x_init=x0, weights=w)
    E, a, x_final, ei, et = res[:5]
    results[tag] = (E, a, x_final, ei, et)
    report_cos_sim(x_final, ei, et, tag)

print("\n" + "=" * 50)
print("\n=== Shake test: low-conflict with vs without shake ===\n")
x_shake = torch.randn(N, D)
res1 = run_simulation('conflict', 'weighted-gd', x_init=x_shake,
    weights={'conflict': 0.1, 'coherence': 1.0, 'support': 1.0})
E_low_noshake, a_low_noshake, x_low_ns, ei_ns, et_ns = res1[:5]
print()
res2 = run_simulation('conflict', 'weighted-gd', x_init=x_shake,
    weights={'conflict': 0.1, 'coherence': 1.0, 'support': 1.0}, shake=True)
E_low_shake, a_low_shake, x_low_s, ei_s, et_s = res2[:5]

report_cos_sim(x_low_ns, ei_ns, et_ns, 'low-conflict no shake')
report_cos_sim(x_low_s, ei_s, et_s, 'low-conflict + shake')

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(NUM_ITERS), E_peace_mp, 'b-', label='Peaceful, MP')
ax1.plot(range(NUM_ITERS), E_conflict_mp, 'r-', label='Conflict, MP')
ax1.plot(range(NUM_ITERS), E_conflict_gd, 'g--', label='Conflict, GD', linewidth=2)

colors = ['m', 'c', 'orange']
for (tag, _), c in zip(weight_sets, colors):
    label_short = tag.replace('  ', '\n')
    ax1.plot(range(NUM_ITERS), results[tag][0], ':', color=c, label=label_short, linewidth=2)
    ax2.plot(range(NUM_ITERS), results[tag][1], ':', color=c, label=label_short, linewidth=2)

ax1.plot(range(NUM_ITERS), E_low_noshake, '-', color='tab:brown', label='low-conflict no shake', linewidth=1)
ax1.plot(range(NUM_ITERS), E_low_shake, '--', color='tab:brown', label='low-conflict + shake', linewidth=2)
ax2.plot(range(NUM_ITERS), a_low_noshake, '-', color='tab:brown', label='low-conflict no shake', linewidth=1)
ax2.plot(range(NUM_ITERS), a_low_shake, '--', color='tab:brown', label='low-conflict + shake', linewidth=2)

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Energy E(x)')
ax1.set_title('Energy convergence')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Iteration')
ax2.set_ylabel(r'$\alpha$')
ax2.set_title(r'$\alpha$ adaptation (meta-regulator)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('egn_convergence.png', dpi=150)
print("\nSaved egn_convergence.png")

# --- Long-run shake analysis ---
print("\n" + "=" * 50)
print("\n=== Long-run shake analysis: low-conflict + shake, 200 steps ===\n")
x_long = torch.randn(N, D)
res_long = run_simulation(
    'conflict', 'weighted-gd', x_init=x_long,
    weights={'conflict': 0.1, 'coherence': 1.0, 'support': 1.0},
    shake=True, num_iters=200)
E_long, a_long, x_long_f, ei_long, et_long, shake_steps, fn_long = res_long

if shake_steps:
    intervals = [shake_steps[i+1] - shake_steps[i] for i in range(len(shake_steps)-1)]
    print(f"\n  Shake step indices: {shake_steps}")
    print(f"  Intervals between shakes: {intervals}")
    print(f"  Min interval: {min(intervals)}, Max interval: {max(intervals)}, Mean interval: {sum(intervals)/len(intervals):.1f}")
    first_half = intervals[:len(intervals)//2]
    second_half = intervals[len(intervals)//2:]
    print(f"  First half mean interval: {sum(first_half)/len(first_half):.1f}")
    print(f"  Second half mean interval: {sum(second_half)/len(second_half):.1f}")
    if len(shake_steps) >= 4:
        last_few = intervals[-3:] if len(intervals) >= 3 else intervals
        print(f"  Last intervals: {last_few}")
    last_shake = shake_steps[-1]
    steps_since_last = 200 - 1 - last_shake
    print(f"  Steps since last shake: {steps_since_last}")
else:
    print("  No shakes triggered.")

report_cos_sim(x_long_f, ei_long, et_long, 'low-conflict + shake, 200 steps')

# --- Gradient norm analysis ---
print("\n" + "=" * 50)
print("\n=== Gradient norm analysis: low-conflict, 100 steps ===\n")
x_grad = torch.randn(N, D)
res_grad = run_simulation('conflict', 'weighted-gd', x_init=x_grad,
    weights={'conflict': 0.1, 'coherence': 1.0, 'support': 1.0},
    num_iters=100)
E_grad, a_grad, x_grad_f, ei_grad, et_grad, ss_grad, fn_grad = res_grad

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(len(fn_grad)), fn_grad, 'b-')
ax1.set_yscale('log')
ax1.set_xlabel('Iteration')
ax1.set_ylabel(r'$||F||$ (log scale)')
ax1.set_title(r'Gradient norm $||\nabla E_{\text{weighted}}||$')
ax1.grid(True, alpha=0.3, which='both')

ax2.plot(range(len(E_grad)), E_grad, 'b-')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Energy E(x)')
ax2.set_title('Energy (linear scale)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('egn_gradient_norm.png', dpi=150)
print("\nSaved egn_gradient_norm.png")

print(f"\n  ||F|| at start:  {fn_grad[0]:.6f}")
print(f"  ||F|| at step 20: {fn_grad[20]:.6f}")
print(f"  ||F|| at step 50: {fn_grad[50]:.6f}")
print(f"  ||F|| at step 99: {fn_grad[99]:.6f}")
print(f"  Final E: {E_grad[-1]:.6f}")
print(f"  Last 10 ||F|| values: {[f'{v:.6f}' for v in fn_grad[-10:]]}")
print(f"  Min ||F|| in last 10: {min(fn_grad[-10:]):.6f}")

# --- Dynamic weights analysis ---
print("\n" + "=" * 50)
print("\n=== Dynamic weights: w_conflict = sigmoid(k*(||g_c||-th)) no shake ===\n")

x_dw = torch.randn(N, D)
res_dw = run_simulation('conflict', 'weighted-gd', x_init=x_dw,
    dynamic_weights=True, dw_k=5.0, dw_threshold=0.15, num_iters=100)
E_dw, a_dw, x_dw_f, ei_dw, et_dw, ss_dw, fn_dw = res_dw

# also track w_conflict trajectory by re-running with logging
print("\n  w_conflict trajectory (every 5 steps):")
print("  step   w_conflict   ||g_c||     E")
x2 = x_dw.clone().requires_grad_(True)
alpha_dw = ALPHA_INIT
w_trace = []
for step in range(100):
    E_c = compute_conflict(x2, ei_dw, et_dw)
    E_coh = compute_coherence(x2)
    E_s = compute_support(x2, ei_dw, et_dw)
    g_c = torch.autograd.grad(E_c, x2, retain_graph=True)[0]
    import math
    g_c_norm = g_c.norm().item()
    wc = 1.0 / (1.0 + math.exp(-5.0 * (g_c_norm - 0.15)))
    w_trace.append(wc)
    g_coh = torch.autograd.grad(E_coh, x2, retain_graph=True)[0]
    g_s = torch.autograd.grad(-E_s, x2, retain_graph=True)[0]
    F = wc * g_c + 1.0 * g_coh + 1.0 * g_s
    with torch.no_grad():
        x2.data -= alpha_dw * F
    if step in [0,5,10,20,30,40,50,70,99]:
        E_val = (E_c + E_coh - E_s).item()
        print(f"  {step:4d}   {wc:.4f}       {g_c_norm:.4f}   {E_val:.4f}")

print(f"\n  Final w_conflict: {w_trace[-1]:.4f}")
print(f"  w_conflict range: [{min(w_trace):.4f}, {max(w_trace):.4f}]")
print(f"  E range: [{min(E_dw):.4f}, {max(E_dw):.4f}]")
print(f"  Final E: {E_dw[-1]:.4f}")
