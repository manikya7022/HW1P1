import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0] 
        self.C = A.shape[1] 
        ln=np.ones((self.N, 1), dtype=float)
        lc=np.ones((self.C, 1), dtype=float)
        se = (A - Y) * (A - Y)  
        sse = ln.T @ se @ lc  
        mse = sse[0,0]/(self.N*self.C) 

        return mse

    def backward(self):

        dLdA = (2*(self.A-self.Y))/(self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        """
        self.A = A
        self.Y = Y
        self.N = Y.shape[0]  
        C = Y.shape[1]  
        self.ln=np.ones((self.N, 1), dtype=float)
        self.lc=np.ones((C, 1), dtype=float)


        exp_A = np.exp(A)
        self.softmax = exp_A / np.sum(exp_A, axis=1, keepdims=True)

        crossentropy = (-Y * np.log(self.softmax))@self.lc 
        sum_crossentropy = self.ln.T @ crossentropy  
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y)/self.N 

        return dLdA
