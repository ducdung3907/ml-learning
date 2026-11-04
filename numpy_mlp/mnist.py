import numpy as np
from sklearn.datasets import fetch_openml
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "MNIST.npz"

def load_mnist(raw=False):
    if CACHE_FILE.exists():
        print("Loading MNIST from local cache...")

        with np.load(CACHE_FILE, allow_pickle=False) as data:
            assert data["X_train"].dtype == np.uint8, "old float cache: delete MNIST.npz and rerun"
            X_train_raw = data["X_train"]
            Y_train = data["Y_train"]
            X_test_raw = data["X_test"]
            Y_test = data["Y_test"]
    else:
        print("Downloading MNIST from OpenML...")

        mnist = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="liac-arff"
        )

        X = mnist.data.astype(np.uint8)
        Y = mnist.target.astype(np.uint8)

        X_train_raw, X_test_raw = X[:60000], X[60000:]
        Y_train, Y_test = Y[:60000], Y[60000:]

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            CACHE_FILE,
            X_train=X_train_raw,
            Y_train=Y_train,
            X_test=X_test_raw,
            Y_test=Y_test
        )

        print("Dataset saved to:", CACHE_FILE)

    if raw:
        return X_train_raw, Y_train, X_test_raw, Y_test

    X_train = X_train_raw.astype(np.float32) / np.float32(255.0)
    X_test = X_test_raw.astype(np.float32) / np.float32(255.0)

    return X_train, Y_train, X_test, Y_test


def one_hot(y, num_classes=10):
    y = np.asarray(y)
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y.astype(np.intp)] = 1
    return encoded


def split_val(X, Y, n_val=5000):
    split_at = len(X) - n_val
    return X[:split_at], Y[:split_at], X[split_at:], Y[split_at:]


def batches(X, Y, batch_size, rng):
    indices = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], Y[batch_indices]


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_mnist()
    print("X_train:", X_train.shape, X_train.dtype)
    print("Y_train:", Y_train.shape, Y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("Y_test:", Y_test.shape, Y_test.dtype)
    Xt, Yt, Xv, Yv = split_val(X_train, Y_train)
    assert len(Xt) == 55000 and len(Xv) == 5000

    oh = one_hot(Yt[:5])
    assert (oh.sum(axis=1) == 1).all() and (oh.argmax(axis=1) == Yt[:5]).all()

    rng = np.random.default_rng(0)
    seen = 0
    for xb, yb in batches(Xt, Yt, 128, rng):
        seen += len(xb)
    assert seen == 55000

    a = next(batches(Xt, Yt, 128, np.random.default_rng(0)))[1]
    b = next(batches(Xt, Yt, 128, np.random.default_rng(0)))[1]
    assert (a == b).all()
