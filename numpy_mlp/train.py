import numpy as np

import mlp
import mnist


def learning_rate(epoch, epochs, lr, schedule="constant", decay=0.95, step_every=10):
	"""Learning rate for a 1-based epoch.

	constant : lr
	exp      : lr * decay ** (epoch - 1)
	step     : lr * decay ** ((epoch - 1) // step_every)
	cosine   : lr anneals to 0 over `epochs` (decay is unused)
	"""
	if schedule == "constant":
		return lr
	if schedule == "exp":
		return lr * decay ** (epoch - 1)
	if schedule == "step":
		return lr * decay ** ((epoch - 1) // step_every)
	if schedule == "cosine":
		return 0.5 * lr * (1 + np.cos(np.pi * (epoch - 1) / epochs))
	raise ValueError(f"unknown lr schedule: {schedule}")


def train(hidden_sizes=(128,), epochs=20, lr=0.1, batch_size=128, seed=0, verbose=True,
		lr_schedule="constant", lr_decay=0.95, lr_step_every=10, optimizer="sgd"):
	X, Y, X_test, Y_test = mnist.load_mnist()
	Xt, Yt, Xv, Yv = mnist.split_val(X, Y)

	rng = np.random.default_rng(seed)
	params = mlp.initialize_parameters(mlp.layer_sizes_for(hidden_sizes), rng)
	opt_state = mlp.init_optimizer(params, optimizer)
	history = []

	for epoch in range(1, epochs + 1):
		epoch_lr = learning_rate(epoch, epochs, lr, lr_schedule, lr_decay, lr_step_every)
		total_loss = 0.0
		for xb, yb in mnist.batches(Xt, Yt, batch_size, rng):
			Yb = mnist.one_hot(yb)
			logits, cache = mlp.forward(params, xb)
			loss, dlogits = mlp.softmax_ce(logits, Yb)
			grads = mlp.backward(params, cache, dlogits, Yb)
			mlp.optimizer_update(params, grads, opt_state, epoch_lr)
			total_loss += loss * len(xb)

		train_loss = total_loss / len(Xt)
		val_acc = mlp.accuracy(params, Xv, Yv)
		train_acc = mlp.accuracy(params, Xt[:5000], Yt[:5000])
		epoch_metrics = {
			"epoch": epoch,
			"lr": epoch_lr,
			"train_loss": train_loss,
			"train_acc": train_acc,
			"val_acc": val_acc,
		}
		history.append(epoch_metrics)

		if verbose:
			print(
				f"epoch {epoch:2d}  lr {epoch_lr:.4f}  loss {train_loss:.3f}  "
				f"train {train_acc:.1%}  val {val_acc:.1%}"
			)

	return params, history


if __name__ == "__main__":
	import time

	start = time.perf_counter()
	params, history = train(hidden_sizes=[128], epochs=20)
	elapsed = time.perf_counter() - start
	np.savez("../data/mlp_h128.npz", **params)
	print(f"Final validation accuracy: {history[-1]['val_acc']:.1%}")
	print(f"Training time: {elapsed:.1f} seconds")

