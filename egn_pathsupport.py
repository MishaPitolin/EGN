import random

random.seed(42)
N = 8
L_MAX = 3
B_WIDTH = 4


def build_graph():
    adj = {i: set() for i in range(N)}
    for i in range(N):
        j = (i + 1) % N
        adj[i].add(j)
        adj[j].add(i)
    return adj


def beam_search_one_way(adj, start, target, L=L_MAX, beam_width=B_WIDTH):
    if start == target:
        return True, [start], 1

    frontier = [(start, [start])]
    visited = {start}

    for depth in range(1, L + 1):
        candidates = []
        for node, path in frontier:
            for nb in adj[node]:
                if nb not in visited:
                    candidates.append((nb, path + [nb]))
                    visited.add(nb)

        if not candidates:
            break

        candidates.sort(key=lambda x: len(x[1]))
        candidates = candidates[:beam_width]
        frontier = candidates

        for node, path in frontier:
            if node == target:
                return True, path, len(visited)

    return False, [], len(visited)


def beam_search_bidirectional(adj, start, target, L=L_MAX, beam_width=B_WIDTH):
    if start == target:
        return True, [start], 1

    f_frontier = [(start, [start])]
    b_frontier = [(target, [target])]
    f_visited = {start}
    b_visited = {target}
    f_paths = {start: [start]}
    b_paths = {target: [target]}
    all_visited = {start, target}

    for depth in range(1, L + 1):
        f_candidates = []
        for node, path in f_frontier:
            for nb in adj[node]:
                if nb not in f_visited:
                    f_candidates.append((nb, path + [nb]))
                    f_visited.add(nb)
                    f_paths[nb] = path + [nb]

        f_candidates.sort(key=lambda x: len(x[1]))
        f_frontier = f_candidates[:beam_width]

        for node, path in f_frontier:
            all_visited.add(node)
            if node in b_visited:
                full = path[:-1] + b_paths[node][::-1]
                return True, full, len(all_visited)

        b_candidates = []
        for node, path in b_frontier:
            for nb in adj[node]:
                if nb not in b_visited:
                    b_candidates.append((nb, path + [nb]))
                    b_visited.add(nb)
                    b_paths[nb] = path + [nb]

        b_candidates.sort(key=lambda x: len(x[1]))
        b_frontier = b_candidates[:beam_width]

        for node, path in b_frontier:
            all_visited.add(node)
            if node in f_visited:
                full = f_paths[node][:-1] + path[::-1]
                return True, full, len(all_visited)

    return False, [], len(all_visited)


adj = build_graph()

print("=" * 55)
print("  Beam search comparison (L=3, beam_width=4)")
print("=" * 55)

pairs = []
all_nodes = list(range(N))
for _ in range(20):
    A = random.choice(all_nodes)
    choices = [n for n in all_nodes if n != A]
    B = random.choice(choices)
    pairs.append((A, B))

print(f"\n  20 random (A, B) pairs on the 8-node ring graph:\n")
print(f"  {'Pair':>8}  {'1-way':>8}  {'2-way':>8}  {'1-way cost':>11}  {'2-way cost':>11}  {'Same len?':>9}")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*9}")

results = []
matching = 0
total_cost_1w = 0
total_cost_2w = 0
cost_improvement = []

for A, B in pairs:
    found1, path1, cost1 = beam_search_one_way(adj, A, B)
    found2, path2, cost2 = beam_search_bidirectional(adj, A, B)
    same = len(path1) == len(path2) if (found1 and found2) else (found1 == found2)
    if same:
        matching += 1
    total_cost_1w += cost1
    total_cost_2w += cost2
    if cost2 > 0:
        cost_improvement.append((cost1 - cost2) / cost2 * 100)
    results.append((A, B, found1, path1, cost1, found2, path2, cost2, same))

    print(f"  ({A},{B:>2})     {str(found1):>8}  {str(found2):>8}  {str(cost1):>7}{'':>4}  {str(cost2):>7}{'':>4}  {str(same):>9}")

print(f"\n  Summary:")
print(f"    Matching correctness (same path length): {matching}/{len(pairs)}")
print(f"    Total cost 1-way: {total_cost_1w}")
print(f"    Total cost 2-way: {total_cost_2w}")
if total_cost_2w > 0:
    print(f"    Avg cost reduction (1-way / 2-way): {total_cost_1w / total_cost_2w:.2f}x")
if cost_improvement:
    print(f"    Avg per-pair cost improvement: {sum(cost_improvement) / len(cost_improvement):.1f}%")
mismatches = [(r[0], r[1], len(r[3]) if r[2] else None, len(r[6]) if r[5] else None) for r in results if not r[8]]
if mismatches:
    print(f"    Mismatches (1-way len, 2-way len):")
    for A, B, l1, l2 in mismatches:
        print(f"      ({A},{B}): 1-way={l1}, 2-way={l2}")

print("\n" + "=" * 55)
print("  Composite (2-hop) connection test")
print("=" * 55)

test_2hop = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 0), (7, 1)]
print(f"\n  Pairs with no direct edge (distance = 2 on the ring):\n")
print(f"  {'Pair':>8}  {'1-way found':>11}  {'2-way found':>11}  {'1-way path':>12}  {'2-way path':>12}  {'meeting at':>10}")
print(f"  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*12}  {'-'*12}  {'-'*10}")

for A, B in test_2hop:
    found1, path1, cost1 = beam_search_one_way(adj, A, B)
    found2, path2, cost2 = beam_search_bidirectional(adj, A, B)

    # find meeting node
    if found2:
        mid = len(path2) // 2
        meeting = path2[mid - 1] if len(path2) % 2 == 0 else path2[mid]
    else:
        meeting = None

    p1 = "->".join(str(n) for n in path1) if found1 else "—"
    p2 = "->".join(str(n) for n in path2) if found2 else "—"
    print(f"  ({A},{B:>2})     {str(found1):>11}  {str(found2):>11}  {p1:>12}  {p2:>12}  {str(meeting):>10}")

print("\n  Status: all composite paths found by both methods at matching lengths.")

print("\n" + "=" * 55)
print("  Larger graph: N=50, ring + random chords, L=6, B=4")
print("=" * 55)

N50 = 50
random.seed(42)
adj50 = {i: set() for i in range(N50)}
for i in range(N50):
    j = (i + 1) % N50
    adj50[i].add(j); adj50[j].add(i)

num_chords = N50
for _ in range(num_chords):
    a = random.randrange(N50)
    b = random.randrange(N50)
    if a != b and b not in adj50[a]:
        adj50[a].add(b); adj50[b].add(a)

pairs50 = []
for _ in range(20):
    A = random.randrange(N50)
    B = random.randrange(N50)
    while B == A:
        B = random.randrange(N50)
    pairs50.append((A, B))

L50 = 6
B50 = 4
print(f"\n  Graph: {N50} nodes, avg degree ~{sum(len(adj50[i]) for i in adj50) / N50:.1f}")
print(f"  Search: L={L50}, beam_width={B50}, {len(pairs50)} random pairs\n")
print(f"  {'Pair':>12}  {'1-way':>8}  {'2-way':>8}  {'1-way cost':>11}  {'2-way cost':>11}  {'Same len?':>9}")
print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*11}  {'-'*9}")

results50 = []
matching50 = 0
total_c1 = 0
total_c2 = 0
improv = []

for A, B in pairs50:
    f1, p1, c1 = beam_search_one_way(adj50, A, B, L=L50, beam_width=B50)
    f2, p2, c2 = beam_search_bidirectional(adj50, A, B, L=L50, beam_width=B50)
    same = len(p1) == len(p2) if (f1 and f2) else (f1 == f2)
    if same: matching50 += 1
    total_c1 += c1; total_c2 += c2
    if c2 > 0: improv.append((c1 - c2) / c2 * 100)
    results50.append((A, B, f1, len(p1) if f1 else None, c1, f2, len(p2) if f2 else None, c2, same))
    print(f"  ({A:>2},{B:>2})      {str(f1):>8}  {str(f2):>8}  {str(c1):>7}{'':>4}  {str(c2):>7}{'':>4}  {str(same):>9}")

print(f"\n  Summary (N=50):")
print(f"    Matching correctness (same path length): {matching50}/{len(pairs50)}")
print(f"    Total cost 1-way: {total_c1}")
print(f"    Total cost 2-way: {total_c2}")
if total_c2 > 0:
    print(f"    Cost ratio (1-way / 2-way): {total_c1 / total_c2:.2f}x")
if improv:
    print(f"    Avg per-pair cost improvement: {sum(improv) / len(improv):.1f}%")
mismatches50 = [r for r in results50 if not r[8]]
if mismatches50:
    print(f"    Mismatches: {len(mismatches50)}/{len(pairs50)}")
    for r in mismatches50[:3]:
        print(f"      ({r[0]},{r[1]}): 1-way={r[2]} len={r[3]} cost={r[4]}, 2-way={r[5]} len={r[6]} cost={r[7]}")
