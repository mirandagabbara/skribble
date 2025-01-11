#This code is for learning about neural networks. This is not used in final project.
import numpy as np
from mnist import MNIST
from sklearn.preprocessing import OneHotEncoder


"""
ReLU activation function
@param x: Neuron layer list
@return: ReLU list
"""
def relu(x):    
    return np.maximum(0, x)

"""
ReLU derivative function for backpropagation
@param x: Neuron layer outputs
@return: ReLU derivative list
"""
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

"""
Softmax activation function
@param x: Neuron layer list
@return: softmax list
"""
def softmax(x):
    x = np.array(x)  # Ensure input is a NumPy array
    exp_values = np.exp(x - np.max(x))  # Stabilize by subtracting max value
    return exp_values / np.sum(exp_values)

"""
Categorical cross-entropy loss function
@param predicted: Predicted outputs
@param actual: Actual outputs
@return: Cross-entropy loss
"""
def categorical_cross_entropy(predicted, actual):
    epsilon = 1e-12  # Avoid log(0)
    predicted = np.clip(predicted, epsilon, 1. - epsilon)
    return -np.sum(actual * np.log(predicted))


class Layer:
    """
    Represents a layer in the neural network.
    """
    def __init__(self, num_inputs, num_neurons, activation_function):
        # Initialize weights and biases for the layer
        self.weights = np.random.randn(num_inputs, num_neurons) * np.sqrt(2. / num_inputs)
        self.biases = np.zeros((1, num_neurons))
        self.activation_function = activation_function
        self.last_inputs = None  # Store inputs for backpropagation
        self.last_outputs = None  # Store outputs for backpropagation

    def forward(self, inputs):
        """
        Perform the forward pass.
        @param inputs: Inputs to the layer
        @return: Outputs from the layer
        """
        self.last_inputs = inputs
        linear_output = np.dot(inputs, self.weights) + self.biases
        self.last_outputs = self.activation_function(linear_output) if self.activation_function else linear_output
        return self.last_outputs

    def backward(self, delta, learning_rate):
        """
        Perform the backward pass.
        @param delta: Gradient from the next layer
        @param learning_rate: Learning rate for weight updates
        @return: Gradient for the previous layer
        """
        if self.activation_function == relu:
            delta *= relu_derivative(self.last_outputs)

        # Compute gradients
        grad_weights = np.dot(self.last_inputs.T, delta)
        grad_biases = np.sum(delta, axis=0, keepdims=True)
        delta_prev = np.dot(delta, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return delta_prev


class NeuralNetwork:
    """
    Represents the neural network.
    """
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs):
        # Create the layers of the network
        self.layers = []
        for i in range(num_hidden_layers):
            input_size = num_inputs if i == 0 else num_hidden_layer_neurons
            self.layers.append(Layer(input_size, num_hidden_layer_neurons, relu))
        self.layers.append(Layer(num_hidden_layer_neurons, num_outputs, softmax))

    def forward(self, inputs):
        """
        Perform the forward pass through the network.
        @param inputs: Inputs to the network
        @return: Network outputs
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, X, y, epochs, learning_rate):
        """
        Train the neural network.
        @param X: Training inputs
        @param y: Training labels (one-hot encoded)
        @param epochs: Number of training epochs
        @param learning_rate: Learning rate for updates
        """
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                inputs = X[i:i+1]
                expected_output = y[i:i+1]

                # Forward pass
                predicted_output = self.forward(inputs)

                # Compute loss
                loss = categorical_cross_entropy(predicted_output, expected_output)
                total_loss += loss

                # Backward pass
                delta = predicted_output - expected_output
                for layer in reversed(self.layers):
                    delta = layer.backward(delta, learning_rate)

            # Print the average loss for this epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X):.4f}")


# Load the MNIST dataset from files
mndata = MNIST('data')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# Convert lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train, dtype=np.int32)
X_test = np.array(X_test)
y_test = np.array(y_test, dtype=np.int32)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

# Create the model with appropriate input and output size
nn = NeuralNetwork(num_inputs=28*28, num_hidden_layers=3, num_hidden_layer_neurons=11, num_outputs=10)

# Train the model in a number of epochs
nn.train(X_train[:1000], y_train_encoded[:1000], epochs=5, learning_rate=0.01)

# Test the model
sample_index = 801
predicted_output = nn.forward(X_test[sample_index:sample_index+1])
print("Predicted:", np.argmax(predicted_output), "Actual:", y_test[sample_index])
print(mndata.display(X_test[sample_index]))
