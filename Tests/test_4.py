import unittest
import numpy as np

from NeuralNetwork import relu, softmax

class TestActivationFunctions(unittest.TestCase):

    def test_relu(self):
        self.assertTrue((relu(np.array([-1, 0, 1, 2])) == np.array([0, 0, 1, 2])).all())
        self.assertTrue((relu(np.array([-3, -2, -1])) == np.array([0, 0, 0])).all())
        self.assertTrue((relu(np.array([3, 4, 5])) == np.array([3, 4, 5])).all())

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        expected_output = np.array([0.09003057, 0.24472847, 0.66524096])
        self.assertTrue(np.allclose(softmax(x), expected_output, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
    print("all tests passed!")