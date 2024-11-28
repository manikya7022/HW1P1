import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """

        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Passing the input through the linear layer followed by the activation layer to get the model output.
        """

        Z0 = self.layers[0].forward(A0) 
        A1 = self.layers[1].forward(Z0)  

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):

        dLdZ0 = self.layers[1].backward(dLdA1)  
        dLdA0 = self.layers[0].backward(dLdZ0)  

        if self.debug:

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Using Relu activations for both the layers.

        """

        self.layers = [Linear(2, 3),ReLU() ,Linear(3,2), ReLU()]  
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        Z0 = self.layers[0].forward(A0)  
        A1 = self.layers[1].forward(Z0)  

        Z1 = self.layers[2].forward(A1)  
        A2 = self.layers[3].forward(Z1)  

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):

        dLdZ1 = self.layers[3].backward(dLdA2)  
        dLdA1 = self.layers[2].backward(dLdZ1) 

        dLdZ0 = self.layers[1].backward(dLdA1)  
        dLdA0 = self.layers[0].backward(dLdZ0)  

        if self.debug:

            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Using ReLU activation function for all the linear layers.)
        """

        self.layers = [Linear(2,4),ReLU(),Linear(4,8),ReLU(),Linear(8,8),ReLU(),Linear(8,4),ReLU(),Linear(4,2),ReLU()] 

        self.debug = debug

    def forward(self, A):
        """
        Passing the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        if self.debug:

            self.A = [A]

        L = len(self.layers)

        for i in range(L):

            A = self.layers[i].forward(A) 

            if self.debug:

                self.A.append(A)

        return A

    def backward(self, dLdA):

        if self.debug:

            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):

            dLdA = self.layers[i].backward(dLdA)  

            if self.debug:

                self.dLdA = [dLdA] + self.dLdA

        return dLdA
