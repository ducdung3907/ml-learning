import numpy as np
 
import mlp
import mnist
 
 
def main(hidden_sizes=(128,), seed=0, n_images=100, steps=200, lr=0.1):
    X_train, Y_train, _, _ = mnist.load_mnist()
    X = X_train[:n_images]
    y = Y_train[:n_images].astype(int)
    Y = mnist.one_hot(y)
 
    params = mlp.initialize_parameters(mlp.layer_sizes_for(hidden_sizes), np.random.default_rng(seed))
 
    initial_loss, _ = mlp.softmax_ce(mlp.forward(params, X)[0], Y)
    print(f"initial loss {initial_loss:.3f}  (ln 10 = {np.log(10):.3f})")
    assert abs(initial_loss - np.log(10)) < 0.5, f"unexpected initial loss: {initial_loss}"
 
    for step in range(steps):
        logits, cache = mlp.forward(params, X)
        loss, probs = mlp.softmax_ce(logits, Y)
        grads = mlp.backward(params, cache, probs, Y)
        mlp.sgd_update(params, grads, lr)
        if step % 50 == 0:
            print(f"  step {step:3d}  loss {loss:.4f}")
 
    final_loss, _ = mlp.softmax_ce(mlp.forward(params, X)[0], Y)
    acc = mlp.accuracy(params, X, y)
    print(f"final loss {final_loss:.4f}, train accuracy {acc:.2%}")
    assert final_loss < 0.1, f"loss did not fall enough: {final_loss}"
    assert acc == 1.0, f"could not memorise {n_images} images"
 
 
if __name__ == "__main__":
    for hidden in [(128,), (64, 32), (64, 32, 16)]:
        print(f"hidden layers {hidden}")
        main(hidden_sizes=hidden)
    print("data checks passed")