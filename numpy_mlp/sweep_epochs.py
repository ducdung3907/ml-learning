"""How many epochs? Train H = 128 once for 50 epochs and look at the curves.

Training is sequential and seeded, so a 10-epoch run is exactly the first 10 epochs
of a 50-epoch run: one long run gives the result for 10, 20, 30, 40 and 50 epochs.

  left  : train loss per epoch
  right : train and validation accuracy per epoch (checkpoints marked)

Run from numpy_mlp/:   python sweep_epochs.py
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt

import train as tr

HIDDEN = [128]         # hidden layer sizes; any number of layers
MAX_EPOCHS = 50
CHECKPOINTS = [10, 20, 30, 40, 50]
LR = 0.1
SEED = 0
OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    t0 = time.perf_counter()
    _, hist = tr.train(hidden_sizes=HIDDEN, epochs=MAX_EPOCHS, lr=LR, seed=SEED, verbose=False)
    print(f"trained {MAX_EPOCHS} epochs at hidden = {HIDDEN} in {time.perf_counter() - t0:.1f} s\n")

    print("epochs | train loss | train acc | val acc")
    for e in CHECKPOINTS:
        m = hist[e - 1]
        print(f"{e:6d} | {m['train_loss']:10.4f} | {m['train_acc']:8.2%}  | {m['val_acc']:.2%}")
    best = max(hist, key=lambda m: m["val_acc"])
    print(f"\nbest validation: {best['val_acc']:.2%} at epoch {best['epoch']}")

    ep = [m["epoch"] for m in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ep, [m["train_loss"] for m in hist])
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax1.set_yscale("log")
    ax1.set_title(f"Train loss (hidden = {HIDDEN})")
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(ep, [100 * m["train_acc"] for m in hist], label="train")
    ax2.plot(ep, [100 * m["val_acc"] for m in hist], label="validation")
    for e in CHECKPOINTS:
        ax2.axvline(e, color="gray", linestyle=":", linewidth=0.8)
    ax2.axvline(best["epoch"], color="red", linestyle="--", linewidth=1, label=f"best val (epoch {best['epoch']})")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(min(100 * hist[0]["val_acc"], 95), 100.2)
    ax2.set_title("Accuracy per epoch")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep_epochs.png"
    fig.savefig(path, dpi=150)
    print("saved", path)
    plt.show()


if __name__ == "__main__":
    main()