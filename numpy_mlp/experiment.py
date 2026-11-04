"""Staged architecture / recipe search. Validation only until the very last step.

Run from numpy_mlp/:   python experiment.py
Results are cached in results/experiment_cache.json, so an interrupted run resumes.

Rules: same seeds everywhere, compare final-epoch validation accuracy, treat differences
under TOL (0.2 points) as ties. Cost = parameter storage in KB (int8: 1 byte per parameter).
Weights are quantized per output column, symmetric int8; activations stay float.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import mlp
import mnist
import train as tr

SEEDS = [0, 1, 2]
FINAL_SEEDS = [0, 1, 2, 3, 4]
TOL = 0.002
OUT = Path(__file__).resolve().parent.parent / "results"
CACHE_PATH = OUT / "experiment_cache.json"

_X, _Y, _Xtest, _Ytest = mnist.load_mnist()
_Xt, _Yt, _Xv, _Yv = mnist.split_val(_X, _Y)

cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def save_cache():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def cost_kb(arch):
    return mlp.count_parameters(mlp.layer_sizes_for(list(arch))) / 1024


def run(arch, epochs, lr, schedule, batch, seed, keep_params=False):
    """Final-epoch validation accuracy for one config and seed (cached)."""
    key = f"{list(arch)}|e{epochs}|lr{lr}|{schedule}|b{batch}|s{seed}"
    if key in cache and not keep_params:
        return cache[key]
    params, hist = tr.train(
        hidden_sizes=list(arch), epochs=epochs, lr=lr, batch_size=batch, seed=seed,
        verbose=False, lr_schedule=schedule, lr_decay=0.5, lr_step_every=max(1, epochs // 4),
    )
    cache[key] = float(hist[-1]["val_acc"])
    save_cache()
    return (cache[key], params) if keep_params else cache[key]


def mean_val(arch, epochs, lr, schedule, batch, seeds=SEEDS):
    accs = [run(arch, epochs, lr, schedule, batch, s) for s in seeds]
    return float(np.mean(accs)), float(np.std(accs))


def quantize_int8(params):
    out = {}
    for k, v in params.items():
        if k.startswith("W"):
            scale = np.maximum(np.abs(v).max(axis=0, keepdims=True), 1e-12) / 127.0
            q = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
            out[k] = (q.astype(np.float32) * scale).astype(np.float32)
        else:
            out[k] = v
    return out


def stage1():
    print("\n== 1. recipe on (64,), 20 epochs ==", flush=True)
    rows = []
    for lr, sched, batch in itertools.product([0.05, 0.1, 0.2, 0.5], ["constant", "step", "cosine"], [64, 128, 256]):
        m, s = mean_val((64,), 20, lr, sched, batch)
        rows.append((m, s, lr, sched, batch))
        print(f"lr {lr:<4} {sched:8} batch {batch:3d}: val {m:.2%} +- {s:.2%}", flush=True)
    rows.sort(reverse=True)
    best = rows[0]
    ties = [r for r in rows[1:] if best[0] - r[0] < TOL]
    print(f"best: lr {best[2]}, {best[3]}, batch {best[4]} -> {best[0]:.2%}")
    print(f"within {TOL:.1%} of best (ties, not separable): {len(ties)} configs")
    for r in ties[:6]:
        print(f"   lr {r[2]}, {r[3]}, batch {r[4]} -> {r[0]:.2%}")
    return best[2], best[3], best[4]


def stage2(lr, sched, batch):
    print("\n== 2. epochs ==", flush=True)
    res = {}
    for e in [10, 20, 30, 40]:
        res[e], s = mean_val((64,), e, lr, sched, batch)
        print(f"{e:2d} epochs: val {res[e]:.2%} +- {s:.2%}", flush=True)
    top = max(res.values())
    chosen = min(e for e, v in res.items() if top - v <= TOL)
    print(f"chosen epochs: {chosen}")
    return chosen


GROUPS = [
    ("3. width only", [(16,), (32,), (64,), (128,), (256,)]),
    ("4. depth, thin first layer", [(32, 32), (32, 16), (32, 32, 32)]),
    ("5. depth at 64", [(64, 32), (64, 64), (64, 32, 16)]),
    ("6. depth at 128", [(128, 32), (128, 64)]),
    ("7. wide second layer", [(32, 128), (64, 128)]),
]


def stages3to7(epochs, lr, sched, batch):
    results = {}
    for title, archs in GROUPS:
        print(f"\n== {title} ==", flush=True)
        for a in archs:
            t0 = time.perf_counter()
            m, s = mean_val(a, epochs, lr, sched, batch)
            results[a] = (m, s)
            print(f"{str(a):12} {cost_kb(a):7.1f} KB  val {m:.2%} +- {s:.2%}  "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return results


def pareto(points):
    """points: {arch: (acc, cost)}. Keep archs with higher accuracy than everything cheaper."""
    front, best = [], -1.0
    for a, (acc, c) in sorted(points.items(), key=lambda kv: (kv[1][1], -kv[1][0])):
        if acc > best:
            front.append(a)
            best = acc
    return front


def stage8(results, epochs, lr, sched, batch):
    print("\n== 8. Pareto, int8, final choice ==", flush=True)
    pts = {a: (m, cost_kb(a)) for a, (m, _) in results.items()}
    front = pareto(pts)
    print("Pareto frontier (cost KB int8 -> val):")
    for a in front:
        print(f"  {str(a):12} {pts[a][1]:7.1f} KB  {pts[a][0]:.2%}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a, (acc, c) in pts.items():
        ax.scatter(c, 100 * acc, color="C0" if a in front else "lightgray", zorder=3)
        ax.annotate(str(a).replace(" ", ""), (c, 100 * acc), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.plot([pts[a][1] for a in front], [100 * pts[a][0] for a in front], color="C0", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("parameter storage, int8 (KB)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Accuracy vs cost (blue = Pareto frontier)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=150)

    top = max(pts[a][0] for a in front)
    knee = next(a for a in front if top - pts[a][0] <= TOL)
    i = front.index(knee)
    finalists = front[max(0, i - 1): i + 2]
    print(f"knee: {knee} (smallest frontier point within {TOL:.1%} of {top:.2%}); finalists {finalists}")

    final = {}
    for a in finalists:
        fp, q, models = [], [], []
        for s in FINAL_SEEDS:
            acc, params = run(a, epochs, lr, sched, batch, s, keep_params=True)
            fp.append(acc)
            q.append(float(mlp.accuracy(quantize_int8(params), _Xv, _Yv)))
            models.append(params)
        final[a] = (float(np.mean(fp)), float(np.mean(q)), models)
        print(f"{str(a):12} {cost_kb(a):7.1f} KB  fp32 val {final[a][0]:.2%}  int8 val {final[a][1]:.2%}  "
              f"(5 seeds, fp32 std {np.std(fp):.2%})", flush=True)

    top_q = max(v[1] for v in final.values())
    winner = min((a for a in final if top_q - final[a][1] <= TOL), key=cost_kb)
    print(f"winner: {winner}  ({cost_kb(winner):.1f} KB int8, {cost_kb(winner) * 4:.1f} KB fp32)")

    # the one and only test-set evaluation
    test_fp = [float(mlp.accuracy(p, _Xtest, _Ytest)) for p in final[winner][2]]
    test_q = [float(mlp.accuracy(quantize_int8(p), _Xtest, _Ytest)) for p in final[winner][2]]
    print(f"TEST {winner}: fp32 {np.mean(test_fp):.2%} +- {np.std(test_fp):.2%}, "
          f"int8 {np.mean(test_q):.2%} +- {np.std(test_q):.2%}  (5 seeds)")
    (OUT / "experiment_final.json").write_text(json.dumps({
        "recipe": {"lr": lr, "schedule": sched, "batch": batch, "epochs": epochs},
        "winner": list(winner), "test_fp32": test_fp, "test_int8": test_q,
    }, indent=1))


def main():
    lr, sched, batch = stage1()
    epochs = stage2(lr, sched, batch)
    print(f"\nrecipe: lr {lr}, {sched}, batch {batch}, {epochs} epochs")
    results = stages3to7(epochs, lr, sched, batch)
    stage8(results, epochs, lr, sched, batch)


if __name__ == "__main__":
    main()
