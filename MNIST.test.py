import numpy as np
import idx2numpy
import time

class Inference:
    def __init__(self):
        pass

    def Run(self,epoch):
        sample_number = 10000   # Number of samples in the test dataset

        test_images, test_labels = self.getTestDataset()    # Getting Test Images and Labels

        parameters = self.getParameters(epoch)   # Getting parameters

        Aj_Epoch,OHE_Epoch,loss_epoch = self.getVariables(sample_number)    
        
        for sample in range(len(test_images)):

            image_sample = test_images[sample].reshape(784,1) # Reshape from (784,) to (784,1)

            OHE,OHE_Epoch = self.getOneHotEncoding(test_labels,sample,OHE_Epoch) # Getting One Hot Encoding (10,1)

            loss_epoch,Aj_Epoch = self.Forward(image_sample,parameters,OHE,loss_epoch,Aj_Epoch,sample) # One Forward Pass

        test_loss = loss_epoch/10000    # Calculating average loss across all samples
        test_accuracy = self.getAccuracy(Aj_Epoch,OHE_Epoch)    # Getting model's accuracy on test dataset

        return test_loss,test_accuracy,Aj_Epoch,OHE_Epoch

#======================================================================================================
# 1. INITIALISATION
#======================================================================================================

    def getTestDataset(self):
        test_images = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
        test_labels = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

        test_images = np.array(test_images.reshape(-1,784),dtype=float)

        test_images = test_images/255


        return test_images,test_labels
    
    def getParameters(self,epoch):
        
        parameters = np.load(f"Test_3.1_LR0-005/parameters_T3.1/parameters_e{epoch}_T3.1.npz")

        parameters_dict = {"Weight_k_i":parameters['wki'],               # weights: input -> hidden (64,784)
                           "Bias_k":parameters['bk'],      # biases:  hidden (64,1)
                           "Weight_j_k":parameters['wjk'],               # weights: hidden -> output (10,64)
                           "Bias_j":parameters['bj']}      # weights: output (10,1)
        return parameters_dict
    
    def getVariables(self,sample_number):
        """Initializing Variables required for plotting.
        Aj_Epoch stores the model's predicted probability for all classes for every iteration,
        OHE_Epoch stores the dataset's True Label for all classes for every iteration,
        norm_grad stores the cumulative total for an epoch
        loss_grad stores the cumulative total for an epoch
        """

        Aj_Epoch = np.zeros((sample_number,10))  # Activation Neuron in Output Layer (10000,10)
        OHE_Epoch = np.zeros((sample_number,10)) # One Hot Encoding (10000,10)
        loss_epoch = 0
        
        return Aj_Epoch,OHE_Epoch,loss_epoch
    
    def getOneHotEncoding(self,labels,sample,OHE_Epoch):
        """Prepares a binary 1 Dimensional array with 10 elements, where each index
        represents its corresponding class. The class of current sample will index
        a 1 for its corresponding index.
        """
        OHE = np.zeros((10,1),dtype=int)

        # Indexing 1 to the index that corresponds to the current digit
        # Returns a 2D array of size (10,1)
        OHE[labels[sample]] = 1 

        OHE_Epoch[sample] = OHE.T

        return OHE,OHE_Epoch

#======================================================================================================
# 2. FORWARD PROPAGATION
#======================================================================================================

    def Forward(self,image_sample,parameters,OHE,loss_epoch,Aj_Epoch,sample): 
        """Runs a forward pass for one sample."""
        
        Ak = self.Activation_k(image_sample,parameters)       # Calculates Activation Neurons in 2nd Layer
        Aj,Aj_Epoch = self.Activation_j(Ak,parameters,Aj_Epoch,sample)   # Calculates Activation Neurons in Output Layer
        loss_epoch = self.CrossEntropyLoss(Aj,OHE,loss_epoch)       # Calculates Loss Value for entire network

        return loss_epoch,Aj_Epoch
    
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
    
    def Activation_k(self,image_sample,parameters):
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
        Zk = (parameters["Weight_k_i"]@image_sample) 

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
        
    def Activation_j(self,Ak,parameters,Aj_Epoch,sample):
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

        Aj_Epoch[sample] = Aj.T # (1,10)

        return Aj,Aj_Epoch

    def CrossEntropyLoss(self,Aj,OHE,loss_epoch):
        """
        Returns loss value for entire network for current sample, which is essentially the
        negative natural log of the model's probability for the True class.
        OHE is in binary format, so only the index of True Class has a value of 1.

        Loss of Network = Sum(OHE[class] * Aj[class]),
        which is the same as:
        Loss of Network = 1 * Aj[True class]

        """
        loss = -np.log(Aj[np.where(OHE == 1)]) # Returns a 1D array of size (1,)

        loss_epoch += loss

        return loss_epoch
#======================================================================================================
# 4. EVALUATION
#======================================================================================================    
    def getAccuracy(self,val_predictions_total,val_true_labels_total):
        """Handles calculating the accuracy of model for 1 epoch from both Training and Validation.
        Model's highest predicted probability for an iteration is considered the model's prediction.
        Compares model's highest predicted probability with true labels of its corresponding index.
        Counts how many model has gotten correct,
        computes the accuracy."""

        test_prediction_indices = np.argmax(val_predictions_total,axis=1)                          # Finds the indices of models prediction for all iterations
        test_sample_indices = np.arange(len(val_true_labels_total))                                # Gets the indices of length of validation dataset
        test_correct = sum(val_true_labels_total[test_sample_indices,test_prediction_indices])          # Computes the total amount model got correct from 1 validation epoch

        test_accuracy = (test_correct/len(val_true_labels_total)) * 100           # Computes accuracy of model from validation
        
        return test_accuracy

    def getCalibration(self,val_predictions_total,val_true_labels_total):
        """Returns models average predicted probability in each bin and model's total correct predictions in each bin. 
        Array of 10 bins, each bin representing a specific predicted probability's range, 
        e.g bin 1 represents predicted probability between 10%-20%,etc. """
                     
        pred_prob_bin = np.zeros((1,10))
        total_bin = np.zeros((1,10))
        correct_bin = np.zeros((1,10))


        max_pred = np.max(val_predictions_total,axis=1)  # Gets all the model's max predictions from each sample, 1D
        bin_index = np.int64(np.floor(max_pred*10))   # Gets each predictions bin index. 1D array where bin index represents its corresponding prediction

        max_pred_indices = np.argmax(val_predictions_total,axis=1)                   # Computing index for model's prediction for each interation
        correct_index = np.array(np.where(val_true_labels_total == 1)[1])         # Computing index for true label
        correct_bin_indices = bin_index[np.where(max_pred_indices == correct_index)] # Computes model's correct predictions in the form of bin number in an array

        #Looping through every bin number
        for bin in range(10):
            pred_prob_bin[0,bin] = np.sum(np.where(bin_index == bin,max_pred,0))  # Storing the summation of model's predicted probability for each bin
            total_bin[0,bin] = np.sum(np.where(bin_index == bin,1,0))             # Storing the summation of total number of samples in each bin
            correct_bin[0,bin] = np.sum(correct_bin_indices == bin)                  # Storing the summation of model's correct predictions for each bin

        
        predicted_accuracy = np.divide(pred_prob_bin , total_bin, out=np.zeros_like(pred_prob_bin) , where = total_bin != 0)  #Computes the average predicted accuracy for each bin
        true_accuracy = np.divide(correct_bin , total_bin, out=np.zeros_like(correct_bin) , where = total_bin != 0)           #Computes the average true accuracy for each bin

        weighted = total_bin/9999 #number of samples in each bin divided by the total number of samples
        difference = np.abs(predicted_accuracy-true_accuracy)
        ECE = weighted@difference.T

        return ECE[0,0]

    def EpochSummary(self,epoch,test_loss,test_acc,ECE):
        """Writing  values to a csv file:
        Epoch Number, Learning Rate, Training Loss, Validation Loss,
        Training Accuracy, Validation Accuracy, Training Time, Validation Time, Average Gradient Magnitude"""

        with open("Test_Summary.csv","a") as f:
            f.write(f"\n{epoch},{test_loss[0]},{test_acc},{ECE}")

        return        
Test = Inference()

for epoch in range(1,51):
    test_loss,test_acc,Aj_Epoch,OHE_Epoch = Test.Run(epoch)
    ECE = Test.getCalibration(Aj_Epoch,OHE_Epoch)
    Test.EpochSummary(epoch,test_loss,test_acc,ECE)