import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    Define 'forward' function
    Define 'backward' function
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))  
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A - (self.A*self.A) 
        dLdZ = dLdA*dAdZ 
        return dLdZ


class Tanh:
    """
    Define 'forward' function
    Define 'backward' function
    """
    def forward(self, Z):
        self.A = np.tanh(Z)  
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A ** 2  
        dLdZ = dLdA * dAdZ  
        return dLdZ


class ReLU:
    """
    Define 'forward' function
    Define 'backward' function
    """
    def forward(self, Z):
        self.Z=Z
        self.A = np.maximum(0, Z) 
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.Z > 0, 1, 0)  
        dLdZ = dLdA * dAdZ 
        return dLdZ

class GELU:
    """
    Define 'forward' function
    Define 'backward' function
    """
    def forward(self, Z):
        self.Z=Z
        self.A = 0.5 * Z * (1 + scipy.special.erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        term1 = 0.5 * (1 + scipy.special.erf(self.Z / np.sqrt(2)))
        term2 = (self.Z / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.Z ** 2)
        dAdZ = term1 + term2
        dLdZ = dLdA * dAdZ  
        return dLdZ


class Softmax:
    """
    Define 'forward' function
    Define 'backward' function
    """

    def forward(self, Z):
        N = Z.shape[0]
        self.A = np.zeros_like(Z) 
        for i in range(N):
            max_zvalue = np.max(Z[i])
            value_of_exp_z = np.exp(Z[i] - max_zvalue) 
            self.A[i] = value_of_exp_z  / np.sum(value_of_exp_z) 
        return self.A

        
    
    def backward(self, dLdA):

        N, C = dLdA.shape 

        dLdZ = np.zeros_like(dLdA) 

   
        for i in range(N):

          
            J = np.zeros((C, C)) 

            for m in range(C):
                for n in range(C):
                    if m==n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    elif m!=n:
                        J[m, n] = -self.A[i, m] * self.A[i, n] 


            dLdZ[i,:] = np.dot(J, dLdA[i]) 

        return dLdZ