"""Sweep the hidden-layer architecture (width and depth) and plot validation accuracy.

  left  : validation accuracy per epoch, one line per architecture
  right : final train vs validation accuracy per architecture (a gap means overfitting)

Run from numpy_mlp/:   python sweep_hidden.py
Choose H by validation accuracy only; test data is not touched here.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt

import mlp
import train as tr

# each entry is the list of hidden layer sizes, so depth and width are both swept
ARCHITECTURES = [[16], [32], [64], [128], [256], [128, 64], [128, 64, 32]]
EPOCHS = 42            # take this from sweep_epochs.py; same for every architecture, so the comparison is fair
LR = 0.1
SEED = 0
OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    runs = {}
    for arch in ARCHITECTURES:
        name = "-".join(map(str, arch))
        t0 = time.perf_counter()
        _, history = tr.train(hidden_sizes=arch, epochs=EPOCHS, lr=LR, seed=SEED, verbose=False)
        runs[name] = history
        last = history[-1]
        print(f"hidden {name:>10}: val {last['val_acc']:.2%}  train {last['train_acc']:.2%}  "
              f"({time.perf_counter() - t0:.1f} s)")

    print("\n    hidden | params  | final val | best val (epoch)")
    for arch, (name, hist) in zip(ARCHITECTURES, runs.items()):
        n_params = mlp.count_parameters(mlp.layer_sizes_for(arch))
        best = max(hist, key=lambda e: e["val_acc"])
        print(f"{name:>10} | {n_params:7d} | {hist[-1]['val_acc']:8.2%}  | "
              f"{best['val_acc']:.2%} ({best['epoch']})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for name, hist in runs.items():
        ax1.plot([e["epoch"] for e in hist], [100 * e["val_acc"] for e in hist],
                 marker="o", markersize=3, label=name)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy (%)")
    ax1.set_title("Validation accuracy per epoch")
    ax1.grid(alpha=0.3)
    ax1.legend(title="hidden sizes")

    names = list(runs)
    ax2.plot(names, [100 * runs[n][-1]["train_acc"] for n in names], marker="s", label="train")
    ax2.plot(names, [100 * runs[n][-1]["val_acc"] for n in names], marker="o", label="validation")
    ax2.tick_params(axis="x", rotation=30)
    ax2.set_xlabel("hidden layer sizes")
    ax2.set_ylabel("accuracy after %d epochs (%%)" % EPOCHS)
    ax2.set_title("Final accuracy vs hidden size")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep_hidden.png"
    fig.savefig(path, dpi=150)
    print("saved", path)
    plt.show()


if __name__ == "__main__":
    main()