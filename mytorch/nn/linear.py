import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        """
        self.W = np.zeros((out_features,in_features))  
        self.b = np.zeros((out_features,1)) 
        
        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        """
        self.A = A
        self.N = A.shape[0]  
        columnmatrix = np.ones((self.N,1))
        Z = A @ (self.W).T + columnmatrix @ (self.b).T
        
        if self.debug:
            print("Forward Pass - Z:\n", Z)
        
        return Z

    def backward(self, dLdZ):

        self.dLdA = dLdZ @ self.W  
        self.dLdW = (dLdZ).T @ self.A  
        ones_N = np.ones((self.N, 1))  
        self.dLdb = (dLdZ).T @ ones_N
        
        if self.debug:
            print("dLdA:", self.dLdA)
            print("dLdW:", self.dLdW)
            print("dLdb:", self.dLdb)

        return self.dLdA
