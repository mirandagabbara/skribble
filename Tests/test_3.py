import numpy as np
from NeuralNetwork import Neuron, Layer, NeuralNetwork


def test_neuron_forward():
    # Test case 1: Known weights and bias
    num_inputs = 3
    neuron = Neuron(num_inputs, activation_function="relu")
    neuron.weights = np.array([0.5, -0.2, 0.1])
    neuron.bias = 0.4
    inputs = np.array([1.0, 2.0, 3.0])
    expected_output = np.dot(neuron.weights, inputs) + neuron.bias
    output = neuron.forward(inputs)
    assert np.isclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test case 2: Zero weights and bias
    neuron.weights = np.zeros(num_inputs)
    neuron.bias = 0.0
    inputs = np.array([1.0, 2.0, 3.0])
    expected_output = 0.0
    output = neuron.forward(inputs)
    assert np.isclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test case 3: Random weights and bias
    neuron.weights = np.random.randn(num_inputs)
    neuron.bias = np.random.randn()
    inputs = np.random.randn(num_inputs)
    expected_output = np.dot(neuron.weights, inputs) + neuron.bias
    output = neuron.forward(inputs)
    assert np.isclose(output, expected_output), f"Expected {expected_output}, but got {output}"


if __name__ == "__main__":
    test_neuron_forward()
    print("All tests passed!")


def test_layer_forward():
    # Test case 1: Known weights and bias for each neuron
    num_inputs = 3
    num_neurons = 2
    layer = Layer(num_inputs, num_neurons)

    # Manually set weights and biases for reproducibility
    layer.neurons[0].weights = np.array([0.5, -0.2, 0.1])
    layer.neurons[0].bias = 0.4
    layer.neurons[1].weights = np.array([-0.3, 0.8, -0.5])
    layer.neurons[1].bias = -0.1

    inputs = np.array([1.0, 2.0, 3.0])

    # Expected outputs from both neurons
    expected_outputs = [
        np.dot(layer.neurons[0].weights, inputs) + layer.neurons[0].bias,
        np.dot(layer.neurons[1].weights, inputs) + layer.neurons[1].bias
    ]

    # Get the actual output
    output = layer.forward(inputs)

    # Check if the outputs match
    assert np.allclose(output, expected_outputs), f"Expected {expected_outputs}, but got {output}"

    # Test case 2: Zero weights and bias for each neuron
    layer.neurons[0].weights = np.zeros(num_inputs)
    layer.neurons[0].bias = 0.0
    layer.neurons[1].weights = np.zeros(num_inputs)
    layer.neurons[1].bias = 0.0

    inputs = np.array([1.0, 2.0, 3.0])
    expected_outputs = [0.0, 0.0]

    output = layer.forward(inputs)
    assert np.allclose(output, expected_outputs), f"Expected {expected_outputs}, but got {output}"


if __name__ == "__main__":
    test_neuron_forward()
    test_layer_forward()
    print("All neuron and layer tests passed!")


def test_neural_network_forward():
    # Test case 1: Known weights and bias for each layer and neuron
    num_inputs = 3
    num_hidden_layers = 2
    num_hidden_layer_neurons = 2
    num_outputs = 1
    network = NeuralNetwork(num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs)

    # Manually set weights and biases for neurons in each layer for reproducibility
    # First hidden layer
    network.layers[0].neurons[0].weights = np.array([0.5, -0.2, 0.1])
    network.layers[0].neurons[0].bias = 0.4
    network.layers[0].neurons[1].weights = np.array([-0.3, 0.8, -0.5])
    network.layers[0].neurons[1].bias = -0.1

    # Second hidden layer
    network.layers[1].neurons[0].weights = np.array([0.2, -0.7])
    network.layers[1].neurons[0].bias = 0.3
    network.layers[1].neurons[1].weights = np.array([-0.4, 0.9])
    network.layers[1].neurons[1].bias = -0.2

    # Output layer
    network.layers[2].neurons[0].weights = np.array([0.6, -0.1])
    network.layers[2].neurons[0].bias = 0.5

    inputs = np.array([1.0, 2.0, 3.0])

    # Manually calculate the expected output
    # First hidden layer output
    hidden_layer_1_output = [
        np.dot(network.layers[0].neurons[0].weights, inputs) + network.layers[0].neurons[0].bias,
        np.dot(network.layers[0].neurons[1].weights, inputs) + network.layers[0].neurons[1].bias
    ]

    # Second hidden layer output
    hidden_layer_2_output = [
        np.dot(network.layers[1].neurons[0].weights, hidden_layer_1_output) + network.layers[1].neurons[0].bias,
        np.dot(network.layers[1].neurons[1].weights, hidden_layer_1_output) + network.layers[1].neurons[1].bias
    ]

    # Output layer output
    expected_output = np.dot(network.layers[2].neurons[0].weights, hidden_layer_2_output) + network.layers[2].neurons[
        0].bias

    # Get the actual output from the network
    output = network.forward(inputs)

    # Check if the final output matches the expected value
    assert np.isclose(output, expected_output), f"Expected {expected_output}, but got {output}"


if __name__ == "__main__":
    test_neuron_forward()
    test_layer_forward()
    test_neural_network_forward()
    print("All neuron, layer, and network tests passed!")
