"""
Deadline 2 experiments (ITITC503 - AI project: Maze and Route Solver)

  E1  Statistics: mean +/- std over 50 instances for all six algorithms
  E2  Weighted A*: speed versus route-quality trade-off
  E3  8-direction movement: effect of the heuristic (zero, Manhattan, Euclidean, Octile)

Run:  python deadline2_experiments.py     (needs maze_route_solver.py in the same folder)
Saves: fig_stats.png, fig_weighted_astar.png, fig_heuristics_8dir.png
"""
import heapq
import math
import random
import statistics as st
import time

from maze_route_solver import ALGOS, WALL, h as manhattan, make_city, make_maze, neighbors, path_cost, search

RUNS = 50


def ms(values):
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


# ---------------------------------------------------------------- E1
def experiment_stats():
    print(f"\nE1: STATISTICS over {RUNS} instances (mean +/- std)")
    data = {}
    for title, maker, n in (("Maze 81x81", make_maze, 81), ("City 60x60", make_city, 60)):
        inst = [maker(n, 100 + k) for k in range(RUNS)]
        opt = [path_cost(g, search(g, s, t, "dijkstra")[0]) for g, s, t in inst]
        print(f"\n{title}")
        print(f"{'algorithm':<18} | {'solved':>6} | {'nodes expanded':>16} | {'path cost':>16} | {'excess %':>8} | {'time ms':>8}")
        print("-" * 88)
        for key, name in ALGOS:
            nodes, costs, ex, times = [], [], [], []
            for (g, s, t), o in zip(inst, opt):
                p, e, sec = search(g, s, t, key)
                nodes.append(e)
                times.append(sec * 1000)
                if p:
                    c = path_cost(g, p)
                    costs.append(c)
                    ex.append(100 * (c - o) / o)
            nm, ns = ms(nodes)
            cm, cs = ms(costs) if costs else (0, 0)
            exm = st.mean(ex) if ex else 0
            data[(title, key)] = (nm, ns, cm, cs)
            print(f"{name:<18} | {len(costs):>3}/{RUNS:<2} | {nm:>8.0f} +/- {ns:<5.0f} | "
                  f"{cm:>8.1f} +/- {cs:<5.1f} | {exm:>8.1f} | {st.mean(times):>8.2f}")
    return data


# ---------------------------------------------------------------- weighted A* (4-direction)
def weighted_astar(g, s, t, w):
    parent, gc = {s: None}, {s: 0}
    pq = [(w * manhattan(s, t), 0, s)]
    closed, expanded = set(), 0
    while pq:
        _, c, p = heapq.heappop(pq)
        if p in closed:
            continue
        closed.add(p)
        expanded += 1
        if p == t:
            path = [t]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1], expanded
        for nb in neighbors(g, p):
            nc = c + g[nb[0]][nb[1]]
            if nb not in gc or nc < gc[nb]:
                gc[nb], parent[nb] = nc, p
                heapq.heappush(pq, (nc + w * manhattan(nb, t), nc, nb))
    return None, expanded


def experiment_weighted():
    print(f"\nE2: WEIGHTED A*  f = g + w*h   ({RUNS} instances)")
    weights = (1.0, 1.5, 2.0, 3.0, 5.0)
    out = {}
    for title, maker, n in (("City 60x60", make_city, 60), ("Maze 81x81", make_maze, 81)):
        inst = [maker(n, 200 + k) for k in range(RUNS)]
        opt = [path_cost(g, search(g, s, t, "dijkstra")[0]) for g, s, t in inst]
        print(f"\n{title}")
        print(f"{'w':>4} | {'nodes expanded':>14} | {'path cost':>9} | {'excess over optimal %':>21} | {'optimal':>7}")
        print("-" * 70)
        for w in weights:
            nodes, costs, ex, nopt = [], [], [], 0
            for (g, s, t), o in zip(inst, opt):
                p, e = weighted_astar(g, s, t, w)
                c = path_cost(g, p)
                nodes.append(e)
                costs.append(c)
                ex.append(100 * (c - o) / o)
                nopt += (c == o)
            print(f"{w:>4} | {st.mean(nodes):>14.0f} | {st.mean(costs):>9.1f} | {st.mean(ex):>21.2f} | {nopt:>3}/{RUNS}")
            out[(title, w)] = (st.mean(nodes), st.mean(ex))
    return out


# ---------------------------------------------------------------- 8-direction movement
SQ2 = math.sqrt(2)
DIRS8 = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
         (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]
HEUR = {
    "Zero (Dijkstra)": lambda a, b: 0.0,
    "Manhattan": lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]),
    "Euclidean": lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
    "Octile": lambda a, b: (max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                            + (SQ2 - 1) * min(abs(a[0] - b[0]), abs(a[1] - b[1]))),
}


def open_grid(n, seed, block=0.25):
    rnd = random.Random(seed)
    while True:
        g = [[WALL if rnd.random() < block else 1 for _ in range(n)] for _ in range(n)]
        g[0][0] = g[n - 1][n - 1] = 1
        if astar8(g, (0, 0), (n - 1, n - 1), HEUR["Zero (Dijkstra)"])[0] is not None:
            return g


def astar8(g, s, t, hf):
    n = len(g)
    gc, closed, expanded = {s: 0.0}, set(), 0
    pq = [(hf(s, t), 0.0, s)]
    while pq:
        _, c, p = heapq.heappop(pq)
        if p in closed:
            continue
        closed.add(p)
        expanded += 1
        if p == t:
            return c, expanded
        for dr, dc, cost in DIRS8:
            r, q = p[0] + dr, p[1] + dc
            if not (0 <= r < n and 0 <= q < n) or g[r][q] == WALL:
                continue
            if dr and dc and (g[p[0] + dr][p[1]] == WALL or g[p[0]][p[1] + dc] == WALL):
                continue                                    # no corner cutting
            nc = c + cost
            if (r, q) not in gc or nc < gc[(r, q)]:
                gc[(r, q)] = nc
                heapq.heappush(pq, (nc + hf((r, q), t), nc, (r, q)))
    return None, expanded


def experiment_8dir():
    print(f"\nE3: 8-DIRECTION MOVEMENT, 60x60 open grid with 25% obstacles ({RUNS} instances)")
    grids = [open_grid(60, 300 + k) for k in range(RUNS)]
    opt = [astar8(g, (0, 0), (59, 59), HEUR["Zero (Dijkstra)"])[0] for g in grids]
    print(f"{'heuristic':<16} | {'nodes expanded':>14} | {'path cost':>9} | {'optimal':>7} | admissible")
    print("-" * 70)
    out = {}
    for name, hf in HEUR.items():
        nodes, costs, nopt = [], [], 0
        for g, o in zip(grids, opt):
            c, e = astar8(g, (0, 0), (59, 59), hf)
            nodes.append(e)
            costs.append(c)
            nopt += abs(c - o) < 1e-9
        adm = "no (overestimates diagonals)" if name == "Manhattan" else "yes"
        print(f"{name:<16} | {st.mean(nodes):>14.0f} | {st.mean(costs):>9.2f} | {nopt:>3}/{RUNS} | {adm}")
        out[name] = st.mean(nodes)
    return out


# ---------------------------------------------------------------- figures
def figures(e1, e2, e3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing - pip install matplotlib to draw the figures")
        return
    names = [n for _, n in ALGOS]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, title in zip(ax, ("Maze 81x81", "City 60x60")):
        a.bar(names, [e1[(title, k)][0] for k, _ in ALGOS], yerr=[e1[(title, k)][1] for k, _ in ALGOS],
              capsize=3, color="#4a7bd0")
        a.set_title(f"{title}: nodes expanded (mean +/- std, {RUNS} runs)")
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("fig_stats.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for title, col in (("City 60x60", "#d04a4a"), ("Maze 81x81", "#4a7bd0")):
        pts = [(e2[(title, w)][0], e2[(title, w)][1], w) for w in (1.0, 1.5, 2.0, 3.0, 5.0)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=title)
        for x, y, w in pts:
            ax.annotate(f"w={w}", (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("nodes expanded (lower = faster)")
    ax.set_ylabel("route cost above optimal (%)")
    ax.set_title("Weighted A*: speed vs route quality")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_weighted_astar.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(list(e3), list(e3.values()), color=["#999", "#d04a4a", "#e0a040", "#3a9a5a"])
    ax.set_ylabel("nodes expanded (mean)")
    ax.set_title("8-direction movement: effect of heuristic")
    plt.tight_layout()
    plt.savefig("fig_heuristics_8dir.png", dpi=130)
    print("\nSaved fig_stats.png, fig_weighted_astar.png, fig_heuristics_8dir.png")


if __name__ == "__main__":
    t0 = time.time()
    e1 = experiment_stats()
    e2 = experiment_weighted()
    e3 = experiment_8dir()
    figures(e1, e2, e3)
    print(f"\nTotal time {time.time() - t0:.0f}s")
