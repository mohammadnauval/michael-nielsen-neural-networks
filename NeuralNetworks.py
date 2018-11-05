import numpy as np

class NeuralNetworks(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]

    def train(self, features, output, epochs, eta):
        for i in range(epochs):
            print(i)
            self.update_mini_batch(features, output, eta)

    def update_mini_batch(self, mini_batch_features, mini_batch_output, eta):
        x = np.asarray(mini_batch_features.transpose())
        y = np.asarray(mini_batch_output.transpose())

        nabla_b, nabla_w = self.backprop(x, y)

        for b, nb in zip(self.biases, nabla_b):
            b -= eta / mini_batch_features * nabla_b

        for w, nw in zip(self.weights, nabla_w):
            w -= eta / mini_batch_features * nabla_w

    def backprop(self, x, y):
        nabla_b = [np.zeros(np.shape(b)) for b in self.biases]
        nabla_w = [np.zeros(np.shape(w)) for w in self.biases]

        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(np.transpose(self.weights[-l + 1]), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def sigmoid(self, x):
        return np.tanh(x)

    def sigmoid_prime(self, x):
        return 1.0 - np.square(np.tanh(x))

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

