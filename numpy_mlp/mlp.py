import numpy as np

def layer_sizes_for(hidden_sizes, n_in=784, n_out=10):
	if isinstance(hidden_sizes, int):
		hidden_sizes = [hidden_sizes]
	return [n_in, *hidden_sizes, n_out]

def count_parameters(layer_sizes):
	return sum(i * o + o for i, o in zip(layer_sizes[:-1], layer_sizes[1:]))

def initialize_parameters(layer_sizes, rng):
	params = {}
	for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), start=1):
		params[f"W{i}"] = (
			rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
		).astype(np.float32)
		params[f"b{i}"] = np.zeros((1, fan_out), dtype=np.float32)
	return params

def forward(params, X):
	L = len(params) // 2
	a = [X]
	z = []
	for layer in range(1, L + 1):
		z_layer = a[layer - 1] @ params[f"W{layer}"] + params[f"b{layer}"]
		if layer < L:
			z.append(z_layer)
			a.append(np.maximum(0, z_layer))
		else:
			logits = z_layer
	cache = {"a": a, "z": z}
	return logits, cache

def softmax_ce(logits, Y_onehot):
	shifted_logits = logits - logits.max(axis=1, keepdims=True)
	exp_logits = np.exp(shifted_logits)
	probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
	loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1))
	return loss, probs

def backward(params, cache, probs, Y_onehot):
	L = len(params) // 2
	batch_size = cache["a"][0].shape[0]
	dz = (probs - Y_onehot) / batch_size
	grads = {}
	for layer in range(L, 0, -1):
		grads[f"W{layer}"] = cache["a"][layer - 1].T @ dz
		grads[f"b{layer}"] = dz.sum(axis=0, keepdims=True)
		if layer > 1:
			da = dz @ params[f"W{layer}"].T
			dz = da * (cache["z"][layer - 2] > 0)
	return grads

def sgd_update(params, grads, lr):
	for name in params:
		params[name] -= lr * grads[name]

def init_optimizer(params, name="sgd"):
	"""Per-parameter state for optimizers that adapt the step from gradient history."""
	zeros = lambda: {k: np.zeros_like(v) for k, v in params.items()}
	if name == "sgd":
		return {"name": name}
	if name == "adagrad":
		return {"name": name, "v": zeros()}
	if name == "rmsprop":
		return {"name": name, "v": zeros()}
	if name == "adam":
		return {"name": name, "m": zeros(), "v": zeros(), "t": 0}
	raise ValueError(f"unknown optimizer: {name}")

def optimizer_update(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
	name = state["name"]
	if name == "sgd":
		sgd_update(params, grads, lr)
	elif name == "adagrad":
		for k in params:
			state["v"][k] += grads[k] ** 2
			params[k] -= lr * grads[k] / (np.sqrt(state["v"][k]) + eps)
	elif name == "rmsprop":
		for k in params:
			state["v"][k] = beta2 * state["v"][k] + (1 - beta2) * grads[k] ** 2
			params[k] -= lr * grads[k] / (np.sqrt(state["v"][k]) + eps)
	elif name == "adam":
		state["t"] += 1
		t = state["t"]
		for k in params:
			state["m"][k] = beta1 * state["m"][k] + (1 - beta1) * grads[k]
			state["v"][k] = beta2 * state["v"][k] + (1 - beta2) * grads[k] ** 2
			m_hat = state["m"][k] / (1 - beta1 ** t)
			v_hat = state["v"][k] / (1 - beta2 ** t)
			params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

def predict(params, X):
	return np.argmax(forward(params, X)[0], axis=1)

def accuracy(params, X, Y):
	return np.mean(predict(params, X) == Y)
