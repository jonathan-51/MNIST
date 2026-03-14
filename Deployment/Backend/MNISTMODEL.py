import numpy as np
import idx2numpy
import time
import matplotlib.pyplot as plt

class MNISTModel:
    def __init__(self):
        self.parameters = np.load("Backend/parameters_e15_T3.1.npz")
        pass

    def predict_model(self,user_input):
        parameters = {"Weight_k_i":self.parameters['wki'],               # weights: input -> hidden (64,784)
                   "Bias_k":self.parameters['bk'],      # biases:  hidden (64,1)
                   "Weight_j_k":self.parameters['wjk'],               # weights: hidden -> output (10,64)
                   "Bias_j":self.parameters['bj']}      # weights: output (10,1)
        
        image = self.ProcessingImage(user_input)
        Aj = self.Forward(image,parameters)
        Aj = np.round(Aj*100,2)
        return Aj
        # return f"{probability:.2f}% Chance That It Is {prediction}"
    
    def ProcessingImage(self,user_input):
        user_input = np.array(user_input.reshape(784,1),dtype=float)

        user_input = user_input/255
        return user_input

    def Forward(self,image,parameters): 
        """Runs a forward pass for one sample."""
        
        Ak = self.Activation_k(image,parameters)       # Calculates Activation Neurons in 2nd Layer
        Aj = self.Activation_j(Ak,parameters)   # Calculates Activation Neurons in Output Layer
        return Aj
    
    def ReLU(self,Zk):
        """
        Introduces non-linearity to network.

        Takes all values of pre activation neurons:
        if value is zero or negative, it will return a 0;
        if value is positive, it will return itself
        """

        # Return a boolean array where values will return true if satisfy RHS of equation, or else it will return false
        Zk_Boolean = Zk > 0 

        #Will apply a mask on Zk, where all indices that are False(zero or negative) will return 0.
        Zk = np.where(Zk_Boolean,Zk,0)

        return Zk
    
    def Activation_k(self,image,parameters):
        """
        Computes value of each 64 activation neurons in the 2nd layer of network in 64 element array.

        Calculates the weighted sum of one pre-activation neuron in the 2nd layer,
        stores it in Zk array with respective index, 
        non-linearility is introduced by applying ReLU function on preactivation neurons
        to compute activation neurons

        Preactivation neurons in hidden layer (2nd layer) --> Zk
        Activation neurons in hidden layer (2nd layer) --> Ak
        """        

        # Matrix Multiplication to calculate weighted sum (64,784)@(784,1)
        # Matrix addition to calculate preactivation neuron (64,1) + (64,1)

        Zk = (parameters["Weight_k_i"]@image) 

        Zk = Zk + parameters["Bias_k"]  

        Ak = self.ReLU(Zk) # Applying ReLU function (64,1)

        return Ak
    
    def SoftMax(self, Zj):
        """
        Applying SoftMax function to Activation Neurons in output layer.

        Returns a 1 Dimensional array with 10 elements representing the model's
        confidence for each class, in probability format, 
        where each index represents its corresponding class
        """
        Zj_nominator = np.exp(Zj - np.max(Zj))   # Computing nominator 
        Zj_denominator = np.sum(Zj_nominator)       # Computing denominator

        Aj = Zj_nominator/Zj_denominator   # Calculates the probability for each possible class

        return Aj
        
    def Activation_j(self,Ak,parameters):
        """
        Computes value of each 10 activation neurons in the output layer of network in a 10 element array.

        Calculates the weighted sum of one pre-activation neuron in the output layer,
        stores it in Zj array with respective index, 
        Softmax function is applied to Zj to compute activation neurons ,
        which returns model's confidence for each class in probability format
     

        Preactivation neurons in output layer (Final layer) --> Zj
        Activation neurons in output layer (Final layer) --> Aj
        """

        # Matrix Multiplication to calculate weighted sum (10,64)@(64,1)
        # Matrix addition to calculate preactivation neuron (10,1)+(10,1)
        Zj = parameters["Weight_j_k"]@Ak
        Zj = Zj + parameters["Bias_j"]

        #Applying Temperature Scaling
        # 
        # Zj = Zj/1.94
        Aj = self.SoftMax(Zj) # Applying Softmax function (10,1)


        return Aj

