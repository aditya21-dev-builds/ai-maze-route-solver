"""
Maze and Route Solver using AI Search Techniques  (ITITC503 - AI)

Algorithms compared on the same state-space model:
  BFS, DFS (uninformed) | Hill climbing, Greedy best-first, A* (informed) | Dijkstra (cost-based)

Experiment 1: mazes with loops  (all moves cost 1)
Experiment 2: city-route grids  (blocked cells + terrain cost 1..5)

Run:  python maze_route_solver.py        (saves search_comparison.png if matplotlib is installed)
"""
import heapq
import random
import time
from collections import deque

WALL = -1


# ---------------------------------------------------------------- generators
def make_maze(n, seed, loop_prob=0.10):
    """n x n grid (n odd). Perfect maze via DFS carving, then knock out some walls
    to create loops so that several different paths exist."""
    rnd = random.Random(seed)
    g = [[WALL] * n for _ in range(n)]
    g[1][1] = 1
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        opts = [(r + dr, c + dc, dr, dc) for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if 0 < r + dr < n - 1 and 0 < c + dc < n - 1 and g[r + dr][c + dc] == WALL]
        if not opts:
            stack.pop()
            continue
        nr, nc, dr, dc = rnd.choice(opts)
        g[r + dr // 2][c + dc // 2] = 1
        g[nr][nc] = 1
        stack.append((nr, nc))
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if g[r][c] == WALL and rnd.random() < loop_prob:
                g[r][c] = 1
    return g, (1, 1), (n - 2, n - 2)


def make_city(n, seed, block_prob=0.25):
    """Open grid with random blocked cells and terrain cost 1..5; guaranteed solvable."""
    rnd = random.Random(seed)
    while True:
        g = [[WALL if rnd.random() < block_prob else rnd.randint(1, 5) for _ in range(n)]
             for _ in range(n)]
        s, t = (0, 0), (n - 1, n - 1)
        g[0][0] = g[n - 1][n - 1] = 1
        if search(g, s, t, "bfs")[0]:
            return g, s, t


# ---------------------------------------------------------------- search
def h(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])      # Manhattan distance (admissible, min cost = 1)


def neighbors(g, p):
    n = len(g)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, c = p[0] + dr, p[1] + dc
        if 0 <= r < n and 0 <= c < n and g[r][c] != WALL:
            yield (r, c)


def build_path(parent, t):
    path = [t]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def path_cost(g, path):
    return sum(g[r][c] for r, c in path[1:])


def search(g, s, t, algo):
    """Returns (path or None, nodes_expanded, seconds)."""
    t0 = time.perf_counter()
    parent = {s: None}
    expanded = 0
    if algo == "bfs":
        q = deque([s])
        while q:
            p = q.popleft()
            expanded += 1
            if p == t:
                return build_path(parent, t), expanded, time.perf_counter() - t0
            for nb in neighbors(g, p):
                if nb not in parent:
                    parent[nb] = p
                    q.append(nb)
    elif algo == "dfs":
        stack = [s]
        seen = {s}
        while stack:
            p = stack.pop()
            expanded += 1
            if p == t:
                return build_path(parent, t), expanded, time.perf_counter() - t0
            for nb in neighbors(g, p):
                if nb not in seen:
                    seen.add(nb)
                    parent[nb] = p
                    stack.append(nb)
    elif algo == "hill":           # steepest-descent on h; no backtracking
        cur = s
        seen = {s}
        while cur != t:
            expanded += 1
            cands = [nb for nb in neighbors(g, cur) if nb not in seen]
            if not cands:
                return None, expanded, time.perf_counter() - t0      # stuck: local minimum / dead end
            best = min(cands, key=lambda x: h(x, t))
            if h(best, t) >= h(cur, t):
                return None, expanded, time.perf_counter() - t0      # no improving neighbour
            parent[best] = cur
            seen.add(best)
            cur = best
        return build_path(parent, t), expanded, time.perf_counter() - t0
    else:                           # greedy / dijkstra / astar share one best-first loop
        def f(p, gc):
            return {"greedy": h(p, t), "dijkstra": gc, "astar": gc + h(p, t)}[algo]
        gcost = {s: 0}
        pq = [(f(s, 0), 0, s)]
        closed = set()
        while pq:
            _, gc, p = heapq.heappop(pq)
            if p in closed:
                continue
            closed.add(p)
            expanded += 1
            if p == t:
                return build_path(parent, t), expanded, time.perf_counter() - t0
            for nb in neighbors(g, p):
                ng = gc + g[nb[0]][nb[1]]
                if algo == "greedy":
                    if nb not in parent:
                        parent[nb] = p
                        heapq.heappush(pq, (f(nb, ng), ng, nb))
                elif nb not in gcost or ng < gcost[nb]:
                    gcost[nb] = ng
                    parent[nb] = p
                    heapq.heappush(pq, (f(nb, ng), ng, nb))
    return None, expanded, time.perf_counter() - t0


# ---------------------------------------------------------------- experiments
ALGOS = [("bfs", "BFS"), ("dfs", "DFS"), ("hill", "Hill climbing"),
         ("greedy", "Greedy best-first"), ("dijkstra", "Dijkstra"), ("astar", "A*")]


def run_experiment(title, sizes, maker, runs=20):
    print(f"\n{title}  (average over {runs} random instances)")
    print(f"{'size':>9} | {'algorithm':<18} | {'solved':>6} | {'expanded':>9} | {'cost':>7} | {'optimal':>7} | {'ms':>7}")
    print("-" * 80)
    table = {}
    for n in sizes:
        inst = [maker(n, seed) for seed in range(runs)]
        # optimal cost from Dijkstra
        opt = [path_cost(g, search(g, s, t, "dijkstra")[0]) for g, s, t in inst]
        for key, name in ALGOS:
            solved = exp = cost = ms = optimal = 0
            for (g, s, t), o in zip(inst, opt):
                path, e, sec = search(g, s, t, key)
                exp += e
                ms += sec * 1000
                if path:
                    solved += 1
                    c = path_cost(g, path)
                    cost += c
                    optimal += (c == o)
            k = max(solved, 1)
            print(f"{n:>4}x{n:<4} | {name:<18} | {solved:>3}/{runs:<2} | {exp / runs:>9.0f} | "
                  f"{cost / k:>7.1f} | {optimal:>3}/{runs:<3} | {ms / runs:>7.2f}")
            table[(n, key)] = (solved / runs, exp / runs, cost / k, optimal / runs, ms / runs)
        print("-" * 80)
    return table


def show_small_maze():
    g, s, t = make_maze(21, 3)
    paths = {k: search(g, s, t, k) for k in ("bfs", "dfs", "astar")}
    print("Sample 21x21 maze, path found by A* ('.' = path, S/G = start/goal):")
    on = set(paths["astar"][0])
    for r in range(21):
        row = ""
        for c in range(21):
            ch = "#" if g[r][c] == WALL else " "
            if (r, c) in on:
                ch = "."
            if (r, c) == s:
                ch = "S"
            if (r, c) == t:
                ch = "G"
            row += ch
        print(row)
    for k, (p, e, _) in paths.items():
        print(f"  {k:<6}: path length {len(p) - 1}, nodes expanded {e}")


def make_figure(maze_table, city_table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figure (pip install matplotlib)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    g, s, t = make_maze(41, 5)
    for a, (key, name) in zip(ax[:2], (("dfs", "DFS"), ("astar", "A*"))):
        path, e, _ = search(g, s, t, key)
        img = [[0.0 if g[r][c] == WALL else 1.0 for c in range(41)] for r in range(41)]
        a.imshow(img, cmap="gray")
        ys, xs = zip(*path)
        a.plot(xs, ys, color="red", lw=1.5)
        a.set_title(f"{name}: path {len(path) - 1}, expanded {e}")
        a.axis("off")
    sizes = sorted({n for n, _ in maze_table})
    w = 0.13
    for i, (key, name) in enumerate(ALGOS):
        ax[2].bar([j + i * w for j in range(len(sizes))],
                  [maze_table[(n, key)][1] for n in sizes], w, label=name)
    ax[2].set_xticks([j + 2.5 * w for j in range(len(sizes))])
    ax[2].set_xticklabels([f"{n}x{n}" for n in sizes])
    ax[2].set_ylabel("nodes expanded (avg)")
    ax[2].set_title("Search effort on mazes")
    ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("search_comparison.png", dpi=130)
    print("Saved search_comparison.png")


if __name__ == "__main__":
    show_small_maze()
    mt = run_experiment("EXPERIMENT 1: mazes with loops", (21, 41, 81), make_maze)
    ct = run_experiment("EXPERIMENT 2: city routes with terrain cost", (20, 40, 60), make_city)
    make_figure(mt, ct)
