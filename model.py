import torch
import torch.nn as nn

# General info:
# Weights: determine the influence of the input to the output, simmilar to the coefficients in the equation.
# Biases: are constant values added to neuron's weighted inputs, acting as an adjustable offset to shift the output.
# During the training, the network learns by adjusting these weights and biases to minimize erros and improve the accuracy of predictions


# NeuralNet class is a blueprint for creating neaural network model. Inherits from the base class nn.Module
class NeuralNet(nn.Module):
    # __init__ method is the constructor where the layers of the network are defined.
    def __init__(self, input_size, hidden_size, num_classes):

        # call the constructor of the parent class, nn.Module (required to make sure everything is set up correctly)
        super(NeuralNet, self).__init__()

        # create linear layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

        # returns x if x>0, otherwise 0
        self.relu = nn.ReLU()

    # Method where the data flows through the network.
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out
