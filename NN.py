import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


#Create a neural network that inherits nn.Module

class NeuralNetwork(nn.Module):

    def __init__(self, input_size, h1, h2, output_size):

        #input size -> hiddenlayer1 -> hiddenlayer2 -> output size

        super().__init__()

        #Pass input size and first hidden layer
        self.fc1 = nn.Linear(input_size, h1)

        #Pass first hidden layer and second hidden layer
        self.fc2 = nn.Linear(h1, h2)

        #Pass second hidden layer and output size
        self.fc3 = nn.Linear(h2, output_size)


    #Function to move everything forward, pass the inputs
    def forward(self, inputs):
        #Rectified linear unit function. 
        #If the number is greater than 0, output number. Else output 0.
        x = F.relu(self.fc1(inputs))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


#Pick a manual seed for randomization
torch.manual_seed(14)

#Create an instance of neural network
NeuralNetwork = NeuralNetwork()

url = ''