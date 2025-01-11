import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from mnist import MNIST
from sklearn.model_selection import train_test_split
import numpy as np



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

model = NeuralNetwork(input_size=28*28, h1=128, h2=64, output_size=10)

# Load in the MNIST dataset 
mndata = MNIST('data')

#X representzs the actual data of each drawing
#Y represents the outcome/prediction
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

#Train Test split
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2, random_state=14)


#Convert X data to float tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train = X_train / 255.0
X_test = X_test / 255.0


#Convert Y labels to tensors long
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

#Set criteorn of model to measure error of predictions
criterion = nn.CrossEntropyLoss()

#Choose optimizer, using Adam optimizer
#Also set learning rate. If error does not decrease after a number of epochs (iterations), lower it
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train the model in a number of epochs
# We send all our data through our netwrok. An epoch is one run of this
epochs = 50
losses = []

for i in range(epochs):
    #Go forward and get a prediction by sending training data
    y_pred = model.forward(X_train) #et predicted results

    #Measure loss (error). Begins high, then decreases
    loss = criterion(y_pred, y_train) #Predicted values vs y_train

    #Track losses
    losses.append(loss.detach().numpy())
    
    #Back propogation take error rate of forward propgation and feed backward through tewrok
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f'Epoch: {i}, Loss: {loss:.4f}')



#Test the model
with torch.no_grad(): #Turn off back propogation
    y_eval = model.forward(X_test) #X_test is test_data, y_eval is predictions
    loss = criterion(y_eval, y_test)
    print(loss)


correct = 0
with torch.no_grad(): #Turn off back propogation
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        #print(f"{i+1}) Prediction: {y_val.argmax().item()}, Actual: {y_test[i]}")

        #Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1
    
#print(f'We got {correct} correct')

