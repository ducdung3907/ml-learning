"""Compare SGD with optimizers that adapt the step size from gradient history.

  left  : validation accuracy per epoch, one line per optimizer
  right : train loss per epoch

Each optimizer gets its own typical learning rate (Adam/RMSprop need ~100x smaller than SGD).
Run from numpy_mlp/:   python sweep_optimizer.py
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt

import train as tr

HIDDEN = [128]
EPOCHS = 20
SEED = 0
TARGETS = [0.96, 0.97, 0.975]
OPTIMIZERS = {          # name -> learning rate
    "sgd": 0.1,
    "adagrad": 0.01,
    "rmsprop": 0.001,
    "adam": 0.001,
}
OUT = Path(__file__).resolve().parent.parent / "results"


def first_epoch_reaching(hist, target):
    for m in hist:
        if m["val_acc"] >= target:
            return m["epoch"]
    return None


def main():
    runs = {}
    for name, lr in OPTIMIZERS.items():
        t0 = time.perf_counter()
        _, hist = tr.train(hidden_sizes=HIDDEN, epochs=EPOCHS, lr=lr, seed=SEED,
                           optimizer=name, verbose=False)
        runs[name] = (hist, time.perf_counter() - t0)

    header = " | ".join(f"val>={t:.1%}" for t in TARGETS)
    print(f"optimizer |    lr | {header} | final val | best val | time")
    for name, (hist, secs) in runs.items():
        reached = " | ".join(f"{str(first_epoch_reaching(hist, t) or '-'):>9}" for t in TARGETS)
        best = max(m["val_acc"] for m in hist)
        print(f"{name:>9} | {OPTIMIZERS[name]:5g} | {reached} | {hist[-1]['val_acc']:8.2%} | "
              f"{best:7.2%} | {secs:.1f} s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (hist, _) in runs.items():
        ep = [m["epoch"] for m in hist]
        ax1.plot(ep, [100 * m["val_acc"] for m in hist], marker="o", markersize=3, label=name)
        ax2.plot(ep, [m["train_loss"] for m in hist], label=name)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy (%)")
    ax1.set_ylim(94, 99)
    ax1.set_title("Validation accuracy per epoch")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("train loss")
    ax2.set_yscale("log")
    ax2.set_title("Train loss per epoch")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep_optimizer.png"
    fig.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
