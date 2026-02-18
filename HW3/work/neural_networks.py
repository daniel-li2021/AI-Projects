"""
CSCI 561 HW3: Neural Networks - MLP for MNIST
Implements: linear_layer, relu, tanh, dropout, SGD with momentum
"""
import numpy as np
import json
import os

# --- Layer Implementations ---

class linear_layer:
    def __init__(self, input_D: int, output_D: int) -> None:
        self.W = np.random.normal(0, 0.1, (output_D, input_D))
        self.b = np.random.normal(0, 0.1, (output_D,))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (batch, input_D) -> (batch, output_D)"""
        return X @ self.W.T + self.b

    def backward(self, X: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """grad: (batch, output_D). Store grad_W, grad_b; return grad w.r.t. X"""
        self.grad_W = grad.T @ X
        self.grad_b = np.sum(grad, axis=0)
        return grad @ self.W

class relu:
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X)

    def backward(self, X: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad * (X > 0)

class tanh:
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    def backward(self, X: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # d/dx tanh(x) = 1 - tanh(x)^2
        return grad * (1 - np.tanh(X) ** 2)

class dropout:
    def __init__(self, r: float):
        self.r = r
        self.mask = None

    def forward(self, X: np.ndarray, is_train: bool) -> np.ndarray:
        if not is_train:
            return X
        self.mask = (np.random.random(X.shape) >= self.r).astype(float) / (1 - self.r)
        return X * self.mask

    def backward(self, X: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return (1.0 / (1 - self.r)) * self.mask * grad

def miniBatchStochasticGradientDescent(model, momentum, _alpha: float, _learning_rate: float) -> None:
    """Update parameters: v = alpha*v - lr*g, w = w + v"""
    for layer in model:
        if hasattr(layer, 'W'):
            if not hasattr(layer, 'velocity_W'):
                layer.velocity_W = np.zeros_like(layer.W)
                layer.velocity_b = np.zeros_like(layer.b)
            if _alpha <= 0:
                layer.W += -_learning_rate * layer.grad_W
                layer.b += -_learning_rate * layer.grad_b
            else:
                layer.velocity_W = _alpha * layer.velocity_W - _learning_rate * layer.grad_W
                layer.velocity_b = _alpha * layer.velocity_b - _learning_rate * layer.grad_b
                layer.W += layer.velocity_W
                layer.b += layer.velocity_b

def softmax(X: np.ndarray) -> np.ndarray:
    exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(z: np.ndarray, y_onehot: np.ndarray) -> float:
    eps = 1e-10
    return -np.sum(y_onehot * np.log(z + eps)) / z.shape[0]

def accuracy(z: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(z, axis=1)
    return np.mean(pred == y)

def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    n = y.shape[0]
    oh = np.zeros((n, K))
    oh[np.arange(n), y.astype(int)] = 1
    return oh

def load_mnist():
    """Load MNIST subset. Uses keras if available, else fallback."""
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    except ImportError:
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X, y = mnist.data.to_numpy(), mnist.target.astype(int).to_numpy()
            X_train, X_test = X[:60000] / 255.0, X[60000:] / 255.0
            y_train, y_test = y[:60000], y[60000:]
        except Exception:
            raise RuntimeError("Need tensorflow or sklearn to load MNIST")
    X_train = X_train.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)
    return X_train, y_train, X_test, y_test

def main(main_params: dict) -> dict:
    np.random.seed(42)
    lr = main_params.get('learning_rate', 0.01)
    momentum = main_params.get('momentum', 0.0)
    dropout_rate = main_params.get('dropout', 0.5)
    act = main_params.get('activation', 'relu')
    n_hidden = main_params.get('n_hidden', 100)
    epochs = main_params.get('epochs', 10)
    batch_size = main_params.get('batch_size', 32)

    X_train, y_train, X_val, y_val = load_mnist()
    n_train, D = X_train.shape
    K = 10

    # Build model: linear -> act -> [dropout] -> linear2 -> softmax
    layer1 = linear_layer(D, n_hidden)
    act_layer = relu() if act == 'relu' else tanh()
    drop_layer = dropout(dropout_rate)
    layer2 = linear_layer(n_hidden, K)

    model = [layer1, act_layer, drop_layer, layer2]
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        X_train, y_train = X_train[perm], y_train[perm]
        train_correct = 0
        for i in range(0, n_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_oh = one_hot(y_batch, K)

            # Forward
            h1 = layer1.forward(X_batch)
            h2 = act_layer.forward(h1)
            h3 = drop_layer.forward(h2, is_train=True)
            a = layer2.forward(h3)
            z = softmax(a)

            train_correct += np.sum(np.argmax(z, axis=1) == y_batch)

            # Backward: softmax + CE gives grad_z = z - y_oh
            grad_z = (z - y_oh) / batch_size
            grad_a = layer2.backward(h3, grad_z)
            grad_h3 = drop_layer.backward(h2, grad_a)
            grad_h2 = act_layer.backward(h1, grad_h3)
            grad_h1 = layer1.backward(X_batch, grad_h2)

            miniBatchStochasticGradientDescent(model, momentum, momentum, lr)

        train_acc = train_correct / n_train
        # Validation (no dropout)
        h1 = layer1.forward(X_val)
        h2 = act_layer.forward(h1)
        h3 = drop_layer.forward(h2, is_train=False)
        z_val = softmax(layer2.forward(h3))
        val_acc = accuracy(z_val, y_val)
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    return history

if __name__ == "__main__":
    params = {
        'learning_rate': 0.01,
        'momentum': 0.0,
        'dropout': 0.5,
        'activation': 'relu',
        'n_hidden': 100,
        'epochs': 10,
        'batch_size': 32
    }
    hist = main(params)
    outfile = "MLP_lr0.01_m0.0_d0.5_arelu.json"
    with open(outfile, 'w') as f:
        json.dump(hist, f, indent=2)
    print(f"Saved to {outfile}")
