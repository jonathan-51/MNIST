import numpy as np
import idx2numpy
import time
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")  

class Convolution:
    def __init__(self):
        pass
    
    def Train(self,train_images,filter_1,filter_2,filter_3):

        feature_map_1 = self.convolution_1(train_images,filter_1)
        pooled_feature_map_1 = self.pooling_1(feature_map_1)
        feature_map_2 = self.convolution_2(pooled_feature_map_1,filter_2)
        pooled_feature_map_2 = self.pooling_2(feature_map_2)
        feature_map_3 = self.convolution_3(pooled_feature_map_2,filter_3)
        pooled_feature_map_3 = self.pooling_3(feature_map_3)
        return pooled_feature_map_3

    def Validation(self):
        _,_,val_images,_ = self.imageLoader()

    def convolution_1(self,image,filter):
        """Handles the first convolution. 
        Passes the 28x28 image through 8 3x3 filters with a stride of 1.
        Padded the image to create a 30x30 image so that the final output after convolution would be a 28x28 image.
        Takes the sum of the 3x3 filter after each scan.
        Returns a (8,28,28) feature map.
        """
        image = np.pad(image,1,mode='constant') # Adding padding to image, won't affect output as padding are zeros
        feature_map = np.zeros((8,28,28))

        for i in range(len(image)-2):
            for k in range(len(image)-2):
                feature_map[:,i,k] = np.sum(filter*image[i:i+3,k:k+3],axis=(1,2))   # Takes the sum of the product of filter and 3x3 of the image.
        return feature_map  

    def pooling_1(self,feature_map):
        """Handles the first pooling.
        Passes the (8,28,28) image through a 2x2 pooling filter with a stride of 2.
        Takes the max value of each 2x2 scan.
        Returns a downsampled (8,14,14) feature map"""

        pooled_feature_map = np.zeros((8,14,14))

        for i,stride_i in enumerate(np.arange(0,28,2)): # Getting indices. i = index of pooled feature map | stride_i = index of feature map during scans.
            for k,stride_k in enumerate(np.arange(0,28,2)): 
                # Scans even indices of feature map and takes the max value of the 2x2 scan.
                pooled_feature_map[:,i,k] = np.max(feature_map[:,stride_i:stride_i+2,stride_k:stride_k+2],axis=(1,2))


        return pooled_feature_map
    
    def convolution_2(self,pooled_feature_map,filter):
        """Handles the second convolution. 
        Passes the (8,14,14) feature map through 16 3x3 filters with a stride of 1, where each filter has 8 channels to scan.
        Padded the image to create a 16x16 feature map so that the final output after convolution would be a 14x14 feature map.
        Takes the sum of the 8 3x3 filter after each scan.
        Returns a (16,14,14) feature map.
        """
        feature_map_2 = np.zeros((16,14,14))
        
        pooled_feature_map = np.pad(pooled_feature_map,((0,0),(1,1),(1,1)),mode='constant') # Padding feature map -> (8,16,16)

        for filter_num in range(16):
            for i in range(len(feature_map_2[0])): # length of 14, i -> 0-13
                for k in range(len(feature_map_2[0])):  # length of 14, i -> 0-13

                    # (3,3) * (8,3,3) = (8,3,3) --> sum(8,3,3) --> (1)
                    feature_map_2[filter_num,i,k] = np.sum(filter[filter_num]*pooled_feature_map[:,i:i+3,k:k+3])

        return feature_map_2
    
    def pooling_2(self,feature_map):
        """Handles the second pooling.
        Passes the (16,14,14) 2nd feature map through a 2x2 pooling filter with a stride of 2.
        Takes the max value of each 2x2 scan.
        Returns a downsampled (16,7,7) feature map"""

        pooled_feature_map_2 = np.zeros((16,7,7))
        for i,stride_i in enumerate(np.arange(0,14,2)): # Getting indices. i = index of pooled feature map | stride_i = index of feature map during scans.
            for k,stride_k in enumerate(np.arange(0,14,2)): # Getting indices. k = index of pooled feature map | stride_k = index of feature map during scans.

                # Each scan scans the even indices of the 16 channels of feature map and takes the max value of the 2x2 scan from each channel forming a (16,) array
                pooled_feature_map_2[:,i,k] = np.max(feature_map[:,stride_i:stride_i+2,stride_k:stride_k+2],axis=(1,2))

        return pooled_feature_map_2
    
    def convolution_3(self,pooled_feature_map,filter):
        """Handles the second convolution. 
        Passes the (16,7,7) pooled feature map through 32 3x3 filters with a stride of 1, where each filter has 16 channels to scan.
        Padded the image to create a 9x9 feature map so that the final output after convolution would be a 7x7 feature map.
        Takes the sum of the 16 3x3 filter after each scan.
        Returns a (32,7,7) feature map.
        """
        feature_map_3 = np.zeros((32,7,7)) 
        
        pooled_feature_map = np.pad(pooled_feature_map,((0,0),(1,1),(1,1)),mode='constant') # Padding feature map -> (8,9,9)

        for filter_num in range(32):
            for i in range(len(feature_map_3[0])): # length of 7, i -> 0-6
                for k in range(len(feature_map_3[0])):  # length of 7, i -> 0-6
                    
                    # Takes the product of a 3x3 scan with a filter for all 16 channels. 
                    # Take the sum of all values within this scan. 
                    # Stores this value in a feature map for the corresponding filter number
                    # (3,3) * (16,3,3) = (16,3,3) --> sum(16,3,3) --> (1)
                    feature_map_3[filter_num,i,k] = np.sum(filter[filter_num]*pooled_feature_map[:,i:i+3,k:k+3])

        return feature_map_3

    def pooling_3(self,feature_map):
        """Handles the third pooling.
        Pad the 3rd feature map to enlarge it to a size of (32,8,8)
        Passes the (32,8,8) 3rd feature map through a 2x2 pooling filter with a stride of 2.
        Takes the max value of each 2x2 scan.
        Returns a downsampled (32,4,4) feature map"""

        pooled_feature_map_3 = np.zeros((32,4,4))
        feature_map = np.pad(feature_map,((0,0),(0,1),(0,1)))

        for i,stride_i in enumerate(np.arange(0,8,2)): # Getting indices. i = index of pooled feature map | stride_i = index of feature map during scans.
            for k,stride_k in enumerate(np.arange(0,8,2)): # Getting indices. k = index of pooled feature map | stride_k = index of feature map during scans.

                # Each scan scans the even indices of the 32 channels of feature map and takes the max value of the 2x2 scan from each channel forming a (32,) array
                pooled_feature_map_3[:,i,k] = np.max(feature_map[:,stride_i:stride_i+2,stride_k:stride_k+2],axis=(1,2))

        return pooled_feature_map_3
    
class MLP:
    def __init__(self):

        pass
    
    def Train(self,MLP_input,sample,parameters,train_labels,Aj_Epoch,OHE_Epoch,loss_epoch):
        """Runs one full training epoch with a sample size of 50000 units.
        Initialisation --> Front Propagation --> Back Propagation
        Updates Parameters after each iteration"""

        # Loops front and back propagation

        train_images_sample = MLP_input.reshape(512,1)  # Reshape from (32,4,4) to (512,1)

        OHE,OHE_Epoch = self.getOneHotEncoding(train_labels,sample,OHE_Epoch)

        Zk,Ak,Aj,loss,Aj_Epoch,loss_epoch = self.Forward(train_images_sample,parameters,OHE,sample,Aj_Epoch,loss_epoch) # One Forward Pass

        return Zk,Ak,Aj,Aj_Epoch,loss_epoch,OHE

#======================================================================================================
# INITIALISATION
#======================================================================================================    

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
# FORWARD PROPAGATION
#======================================================================================================

    def Forward(self,image_sample,parameters,OHE,sample,Aj_Epoch,loss_epoch): 
        """Runs a forward pass for one sample."""
        
        Ak, Zk = self.Activation_k(image_sample,parameters)       # Calculates Activation Neurons in 2nd Layer
        Aj,Aj_Epoch = self.Activation_j(Ak,parameters,sample,Aj_Epoch)   # Calculates Activation Neurons in Output Layer
        loss,loss_epoch = self.CrossEntropyLoss(Aj,OHE,loss_epoch)       # Calculates Loss Value for entire network

        return Zk,Ak,Aj,loss,Aj_Epoch,loss_epoch
    
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
        image_sample = image_sample/np.max(image_sample)
        Zk = (parameters["Weight_k_i"]@image_sample) 
        Zk = Zk + parameters["Bias_k"]  

        Ak = self.ReLU(Zk) # Applying ReLU function (64,1)

        return Ak, Zk
    
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
        
    def Activation_j(self,Ak,parameters,sample,Aj_Epoch):
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
        # Storing all Activating neurons (50000,10)
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

        return loss[0],loss_epoch

#======================================================================================================
# BACKWARD PROPAGATION
#======================================================================================================    
class Backward:
    def __init__(self):
        pass

    def Backward(self,Aj,Ak,OHE,parameters,Zk,MLP_input,LR,norm_grad):
        """Runs a Backward Pass for one sample"""
        MLP_input = MLP_input.reshape(512,1)  # Reshape from (32,4,4) to (512,1)
        gradients,norm_grad = self.getGradientsMLP(Aj,Ak,OHE,parameters,Zk,MLP_input,norm_grad)  #Computes gradient values for all parameters
    
        updated_parameters = self.ParametersUpdate(parameters,gradients,LR) #Returns updated parameters
        return updated_parameters,norm_grad
    
    def getGradientsConv(self):
        
        return

    def dReLU(self,Zk):
        """Returns the derivatives of all Ak values w.r.t their respective Zk values"""

        indices = Zk > 0    #Returns all indices where its values are positive as Boolean Type

        Zk = np.where(indices,1,0)  #Applies index mask, where all True Values become 1, and all False values become 0

        return Zk
    
    def getGradientsMLP(self,Aj,Ak,OHE,parameters,Zk,MLP_input,norm_grad):
        """Calculates gradients for all parameters.
        Returns a dictionary of all 4 types of parameters, where each gradient value
        represents the value of the same index"""
        
        dZ_j = Aj - OHE #Preactivation Neurons in Output Layer
        dZ_k = ((parameters["Weight_j_k"].T)@dZ_j)*self.dReLU(Zk)   #Preactivation Neurons in Hidden Layer
        

        db_j = dZ_j #Biases in Output Layer
        dW_jk = dZ_j@Ak.T   #Weights in Hidden Layer
        
        
        db_k = dZ_k #Biases in Hidden Layer
        dw_ki = dZ_k@MLP_input.T  #Weights in Input Layer

        gradients = {"Weight_k_i":dw_ki,
                     "Bias_k":db_k,
                     "Weight_j_k":dW_jk,
                     "Bias_j":db_j}
 
        # Calculating magnitude all gradients from each parameter for every sample
        norm_grad += np.sqrt((np.sum(dW_jk**2)) + (np.sum(db_j**2)) + (np.sum(dw_ki**2)) + (np.sum(db_k**2)))

        return gradients,norm_grad

    def ParametersUpdate(self,parameters,gradients,LR):
        """Updating parameters by taking the difference between its value and 
        its gradient multiplied by factor (Learning Rate).
        Storing the updated parameters in a dictionary."""

        weight_k_i_updated = parameters["Weight_k_i"] - LR * gradients["Weight_k_i"]    #Weights in Input Layer
        bias_k_updated = parameters["Bias_k"] - LR * gradients["Bias_k"]                #Biases in Hidden Layer

        weight_j_k_updated = parameters["Weight_j_k"] - LR * gradients["Weight_j_k"]    #Weights in Hidden Layer Layer
        bias_j_updated = parameters["Bias_j"] - LR * gradients["Bias_j"]                #Biases in Output Layer

        updated_parameters = {"Weight_k_i":weight_k_i_updated,
                              "Bias_k":bias_k_updated,
                              "Weight_j_k":weight_j_k_updated,
                              "Bias_j":bias_j_updated}

        return updated_parameters

#======================================================================================================
# RUN
#======================================================================================================   

class Model:
    def __init__(self,model_conv,model_MLP,model_Backward):
        self.model_conv = model_conv
        self.model_MLP = model_MLP
        self.model_Backward = model_Backward
        pass

    def Run(self,e,LR):
        sample_number = 1
        train_images,train_labels,validation_images,validation_labels = self.imageLoader()
        Aj_Epoch,OHE_Epoch,norm_grad,loss_epoch = self.getVariables(sample_number)  # Initilializing variables
        filter_1,filter_2,filter_3 = self.getParameters()
        parameters = self.getTrainingParameters(e) 

        for sample in range(sample_number):
            train_images_sample = train_images[sample]
            MLP_input = self.model_conv.Train(train_images_sample,filter_1,filter_2,filter_3)
            Zk,Ak,Aj,Aj_Epoch,loss_epoch,OHE = self.model_MLP.Train(MLP_input,sample,parameters,train_labels,Aj_Epoch,OHE_Epoch,loss_epoch)
            updated_parameters,norm_grad = self.model_Backward.Backward(Aj,Ak,OHE,parameters,Zk,MLP_input,LR,norm_grad)
            print(updated_parameters["Weight_k_i"].shape)
            print(updated_parameters["Bias_k"].shape)
            print(updated_parameters["Weight_j_k"].shape)
            print(updated_parameters["Bias_j"].shape)
        return
    
#======================================================================================================
# INITIALISATION
#======================================================================================================  

    def imageLoader(self):
        images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
        labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")    

        images = images/255

        train_images = images[:50000]
        train_labels = labels[:50000]
        validation_images = images[50000:]
        validation_labels = labels[:50000]

        return train_images,train_labels,validation_images,validation_labels

    def getParameters(self):
        """Loading parameters"""

        #============Filter: Convolution 1============#
        
        filter_1 = np.random.normal(loc=0,scale=np.sqrt((2/72)),size=(8,3,3))

        #============Filter: Convolution 2============#

        filter_2 = np.random.normal(loc=0,scale=np.sqrt((2/144)),size=(16,3,3))

        #============Filter: Convolution 3============#

        filter_3 = np.random.normal(loc=0,scale=np.sqrt((2/288)),size=(32,3,3))

        return filter_1,filter_2,filter_3

    def getTrainingParameters(self,e):
        """Loads pre-trained model weights and biases from file."""
        
        #parameters = numpy.load(f"Test_3.1_LR0-005/parameters_T3.1/parameters_e{e-1}_T3.2.npz")
        parameters = np.load(f"Convolution/initial_parameters_512.npz")
        #parameters = numpy.load(f"initial_parameters.npz")
        parameters_dict = {"Weight_k_i":parameters['wki'],               # weights: input -> hidden (64,784)
                            "Bias_k":parameters['bk'].reshape(64,1),      # biases:  hidden (64,1)
                            "Weight_j_k":parameters['wjk'],               # weights: hidden -> output (10,64)
                            "Bias_j":parameters['bj'].reshape(10,1)}      # weights: output (10,1)

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
        norm_grad = 0  
        loss_epoch = 0
        
        return Aj_Epoch,OHE_Epoch,norm_grad,loss_epoch

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
# 4. EVALUATION
#======================================================================================================  
class Evaluation:
    def __init__(self,model):
        self.model = model

        pass

    def getEpochStats(self):
        """Aggregate all data required to log for epoch summary."""

        train_loss_avg,val_loss_avg = self.getLoss()    # gets average loss for both training and validation for one epoch

        train_acc,val_acc = self.getAccuracy()  #gets model's correct predictions for both training and validation

        # gets model's training time and validation time for one epoc
        train_time = self.model.traintime   
        val_time = self.model.valtime

        norm_grad_mean = self.model.normgrad    # gets normalized gradient for one epoch (Average magnitude of all parameters gradient from one iteration)

        return train_loss_avg,val_loss_avg,train_acc,val_acc,train_time,val_time,norm_grad_mean

model_conv = Convolution()
model_MLP = MLP()
model_Backward = Backward()
model = Model(model_conv,model_MLP,model_Backward)
model.Run(e=1,LR=0.005)


