import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def drelu(x):
    return (x > 0).astype(int)


def compute_loss(yhat, y, num_samples):
    return (1 / num_samples) * (-(np.dot(y, np.log(yhat).T) + np.dot((1 - y), np.log(1 - yhat).T)))


def forward_pass(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def backward_prop(x, y, a1, a2, w1, w2, b1, b2, m):
    delta_2 = a2 - y
    dW2 = (1 / m) * np.dot(delta_2, a1.T)
    db2 = (1 / m) * np.sum(delta_2, axis=1, keepdims=True)
    delta_1 = np.dot(w2.T, delta_2) * drelu(a1)
    dW1 = (1 / m) * np.dot(delta_1, x.T)
    db1 = (1 / m) * np.sum(delta_1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(w1, w2, b1, b2, dl_dw1, dl_dw2, dl_db1, dl_db2, learning_rate):
    w1 = w1 - (learning_rate * dl_dw1)
    w2 = w2 - (learning_rate * dl_dw2)
    b1 = b1 - (learning_rate * dl_db1)
    b2 = b2 - (learning_rate * dl_db2)
    return w1, w2, b1, b2


def predictions(arr):
    return np.argmax(arr, axis=0)


def get_accuracy(yhat, y):
    return np.sum(yhat == y) / y.size


def gradient_descent(x, y, iterations, num_samples):
    input_size = x.shape[0]
    hidden_size = 5
    output_size = 2

    w1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

    for i in range(iterations):
        z1, a1, z2, a2 = forward_pass(x, w1, b1, w2, b2)
        lw1, lb1, lw2, lb2 = backward_prop(x, y, a1, a2, w1, w2, b1, b2, num_samples)
        w1, w2, b1, b2 = update_params(w1, w2, b1, b2, lw1, lw2, lb1, lb2, 0.01)

        if i % 10 == 0:
            prediction = predictions(a2)
            print(f'Epoch: {i}, Accuracy: {get_accuracy(prediction, y)}')


# # Generate dummy data
x_train = np.random.rand(2, 100)
y_train = np.random.randint(0, 2, 100)
# Model to learn the XOR truth table 
# x_train= np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # XOR input
# y_train= np.array([[0, 1, 1, 0]]) # XOR output

gradient_descent(x_train, y_train, 200, 100)
