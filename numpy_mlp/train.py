import numpy as np

import mlp
import mnist


def train(hidden_size=128, epochs=20, lr=0.1, batch_size=128, seed=0, verbose=True):
	X, Y, X_test, Y_test = mnist.load_mnist()
	Xt, Yt, Xv, Yv = mnist.split_val(X, Y)

	rng = np.random.default_rng(seed)
	params = mlp.initialize_parameters([784, hidden_size, 10], rng)
	history = []

	for epoch in range(1, epochs + 1):
		total_loss = 0.0
		for xb, yb in mnist.batches(Xt, Yt, batch_size, rng):
			Yb = mnist.one_hot(yb)
			logits, cache = mlp.forward(params, xb)
			loss, dlogits = mlp.softmax_ce(logits, Yb)
			grads = mlp.backward(params, cache, dlogits, Yb)
			mlp.sgd_update(params, grads, lr)
			total_loss += loss * len(xb)

		train_loss = total_loss / len(Xt)
		val_acc = mlp.accuracy(params, Xv, Yv)
		train_acc = mlp.accuracy(params, Xt[:5000], Yt[:5000])
		epoch_metrics = {
			"epoch": epoch,
			"train_loss": train_loss,
			"train_acc": train_acc,
			"val_acc": val_acc,
		}
		history.append(epoch_metrics)

		if verbose:
			print(
				f"epoch {epoch:2d}  loss {train_loss:.3f}  "
				f"train {train_acc:.1%}  val {val_acc:.1%}"
			)

	return params, history


if __name__ == "__main__":
	import time

	start = time.perf_counter()
	params, history = train(hidden_size=128, epochs=20)
	elapsed = time.perf_counter() - start
	np.savez("../data/mlp_h128.npz", **params)
	print(f"Final validation accuracy: {history[-1]['val_acc']:.1%}")
	print(f"Training time: {elapsed:.1f} seconds")

