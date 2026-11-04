import numpy as np

def initialize_parameters(layer_sizes, rng):
	params = {}
	for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), start=1):
		params[f"W{i}"] = (
			rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
		).astype(np.float32)
		params[f"b{i}"] = np.zeros((1, fan_out), dtype=np.float32)
	return params

def forward(params, X):
	z1 = X @ params["W1"] + params["b1"]
	a1 = np.maximum(0, z1)
	logits = a1 @ params["W2"] + params["b2"]
	cache = {"X": X, "z1": z1, "a1": a1}
	return logits, cache

def softmax_ce(logits, Y_onehot):
	shifted_logits = logits - logits.max(axis=1, keepdims=True)
	exp_logits = np.exp(shifted_logits)
	probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
	loss = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1))
	return loss, probs

def backward(params, cache, probs, Y_onehot):
	X = cache["X"]
	z1 = cache["z1"]
	a1 = cache["a1"]
	batch_size = X.shape[0]

	dz2 = (probs - Y_onehot) / batch_size
	dW2 = a1.T @ dz2
	db2 = dz2.sum(axis=0, keepdims=True)
	da1 = dz2 @ params["W2"].T
	dz1 = da1 * (z1 > 0)
	dW1 = X.T @ dz1
	db1 = dz1.sum(axis=0, keepdims=True)

	return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def sgd_update(params, grads, lr):
	for name in params:
		params[name] -= lr * grads[name]

def predict(params, X):
	return np.argmax(forward(params, X)[0], axis=1)

def accuracy(params, X, Y):
	return np.mean(predict(params, X) == Y)
