import numpy as np
from mnist import MNIST


"""
ReLU activation function
@param x: Neuron layer list
@return: ReLU list
"""
def relu(x):
    relu_list = []
    for output in range(len(x)):
        relu_list.append(np.maximum(0, x[output]))
    return relu_list

"""
Softmax activation function
@param x: Neuron layer list
@return: softmax list 
"""
def softmax(x):
    exp_values = []
    for output in x:
        exp_values.append(np.exp(output))

    norm_base = sum(exp_values)

    norm_values = []
    for value in exp_values:
        norm_values.append(value / norm_base)

    return norm_values



def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def categorical_cross_entropy(predicted, actual):
    epsilon = 1e-12  # To avoid log(0)
    predicted = np.clip(predicted, epsilon, 1. - epsilon)
    return -np.sum(actual * np.log(predicted))


class Neuron:
    def __init__(self, num_inputs, activation_function):
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = np.random.randn(num_inputs) * np.sqrt(2. / num_inputs)  # He initialization
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.d_weights = None
        self.d_bias = None
        self.delta = 0

    def forward(self, inputs):
        self.output = 0
        self.output = np.dot(self.weights, inputs) + self.bias

        #Apply the ReLU activation function
        if self.activation_function is not None:
            self.output = self.activation_function(self.output)

        return self.output

    def backward(self, delta, learning_rate):
        if self.activation_function == "relu":
            delta *= relu_derivative(self.output)

        self.d_weights = delta * self.inputs
        self.d_bias = delta

        self.delta = delta
        return np.dot(delta, self.weights)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias


class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    # Your forward function goes here before backwardq

        self.neurons = []  # Initialize an empty list to store neurons
        for i in range(num_neurons):
            neuron = Neuron(num_inputs, activation_function)  # Create a new Neuron object
            self.neurons.append(neuron)  # Add the neuron to the list

        self.outputs = []

    def forward(self, inputs):
        self.outputs = []
        # Take the inputs and pass them each neuron's forward functions and store the outputs in the self.outputs list
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(inputs))

        self.outputs = self.activation_function(self.outputs)
        return self.outputs


    def backward(self, delta, learning_rate):
        next_delta = np.zeros(self.num_inputs)
        for i, neuron in enumerate(self.neurons):
            next_delta += neuron.backward(delta[i], learning_rate)
        return next_delta

    def update(self, learning_rate):
        for neuron in self.neurons:
            neuron.update(learning_rate)


class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_outputs = num_outputs

        # Now that we have all the required variables, go ahead and create the layers
        # Always remember that we do NOT need to create a layer for the inputs. The initial
        # inputs that we get make up the first input layer. So, we start from the first hidden
        # layer and create layers all the way up to the last (output) layer
        self.layers = []

        for i in range(num_hidden_layers):
            if i == 0:
                input_size = num_inputs
            else:
                input_size = self.num_hidden_layer_neurons

            hidden_layer = Layer(input_size, self.num_hidden_layer_neurons, relu)

            self.layers.append(hidden_layer)

        # At the end, create the output layer
        output_layer = Layer(self.num_hidden_layer_neurons, self.num_outputs, softmax)
        self.layers.append(output_layer)



    def forward(self, inputs):
        # Take the inputs and pass those inputs to each layer in the network
        output = inputs

        # Keep updating that single variable with the outputs of the layers
        for layer in self.layers:
            output = layer.forward(output)

        # At the end, whatever is in that variable will be the output of the last layer
        return output

    def calc_loss_delta(self, predicted_outputs, actual_outputs):
        return predicted_outputs - actual_outputs

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(X.shape[0]):
                inputs = X[i]
                expected_output = y[i]

                # Forward pass
                predicted_output = self.forward(inputs)

                # Calculate loss
                loss = categorical_cross_entropy(predicted_output, expected_output)
                total_loss += loss

                # Calculate loss gradient (delta for the output layer)
                loss_delta = self.calc_loss_delta(predicted_output, expected_output)

                # Backward pass
                delta = loss_delta
                for layer in reversed(self.layers):
                    delta = layer.backward(delta, learning_rate)
                    layer.update(learning_rate)

            average_loss = total_loss / X.shape[0]
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')



mndata = MNIST('data')



X_train, y_train = mndata.load_training()

X_test, y_test = mndata.load_testing()



# Convert lists to NumPy arrays

X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



# Normalize the data

X_train = X_train / 255.0

X_test = X_test / 255.0
