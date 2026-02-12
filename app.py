import numpy as np
import matplotlib.pyplot as plt
import random

# ================== LOAD DATA ==================
X_train = np.loadtxt('train_X.csv', delimiter=',').T
Y_train = np.loadtxt('train_label.csv', delimiter=',').T

X_test = np.loadtxt('test_X.csv', delimiter=',').T
Y_test = np.loadtxt('test_label.csv', delimiter=',').T

print("shape of X_train : ", X_train.shape)
print("shape of Y_train : ", Y_train.shape)
print("shape of X_test : ", X_test.shape)
print("shape of Y_test : ", Y_test.shape)

# Show random training image
index = random.randrange(0, X_train.shape[1])
plt.imshow(X_train[:, index].reshape((28, 28)), cmap='gray')
plt.title("Random Training Image")
plt.show()

# ================== ACTIVATIONS ==================
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    expX = np.exp(x - np.max(x, axis=0, keepdims=True))
    return expX / np.sum(expX, axis=0, keepdims=True)

def der_relu(x):
    return np.array(x > 0, dtype=np.float32)

# ================== INITIALIZE ==================
def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

# ================== FORWARD ==================
def forward_prop(x, parameters):
    w1, b1 = parameters['w1'], parameters['b1']
    w2, b2 = parameters['w2'], parameters['b2']

    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

# ================== COST ==================
def cost_func(a2, y):
    m = y.shape[1]
    return -(1/m) * np.sum(y * np.log(a2 + 1e-8))

# ================== BACKWARD ==================
def backward_prop(x, y, parameters, cache):
    m = x.shape[1]

    w2 = parameters['w2']
    a1, a2 = cache['a1'], cache['a2']

    dz2 = a2 - y
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * der_relu(a1)
    dw1 = (1/m) * np.dot(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}

# ================== UPDATE ==================
def update_parameters(parameters, grads, lr):
    parameters['w1'] -= lr * grads['dw1']
    parameters['b1'] -= lr * grads['db1']
    parameters['w2'] -= lr * grads['dw2']
    parameters['b2'] -= lr * grads['db2']
    return parameters

# ================== MODEL ==================
def model(x, y, n_h, learning_rate, iterations):
    n_x = x.shape[0]
    n_y = y.shape[0]

    parameters = initialize_parameters(n_x, n_h, n_y)
    cost_list = []

    for i in range(iterations):
        cache = forward_prop(x, parameters)
        cost = cost_func(cache['a2'], y)
        grads = backward_prop(x, y, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        cost_list.append(cost)

        if i % (iterations // 10) == 0:
            print("Cost after iteration", i, ":", cost)

    return parameters, cost_list

# ================== ACCURACY ==================
def accuracy(x, y, parameters):
    cache = forward_prop(x, parameters)
    a2 = cache['a2']
    pred = np.argmax(a2, axis=0)
    labels = np.argmax(y, axis=0)
    return np.mean(pred == labels) * 100

# ================== TRAIN ==================
iterations = 100
n_h = 1000
parameters, cost_list = model(X_train, Y_train, n_h, learning_rate=0.01, iterations=iterations)

print("Training Finished ✅")

# ================== GRAPH ==================
plt.plot(range(len(cost_list)), cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Training Cost Curve")
plt.show()

# ================== ACCURACY RESULT ==================
print("Train Accuracy:", accuracy(X_train, Y_train, parameters), "%")
print("Test Accuracy :", accuracy(X_test, Y_test, parameters), "%")

# ================== RANDOM TEST PREDICTION ==================
idx = random.randrange(0, X_test.shape[1])
plt.imshow(X_test[:, idx].reshape(28, 28), cmap='gray')
plt.title("Test Image")
plt.show()

cache = forward_prop(X_test[:, idx].reshape(784, 1), parameters)
pred = np.argmax(cache['a2'], axis=0)
print("Model Prediction:", pred[0])