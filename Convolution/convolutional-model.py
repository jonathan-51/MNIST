import numpy as np
import idx2numpy
import time
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class convModel:
    def __init__(self):
        pass
    
    def run(self):
        images,labels = self.imageLoader()

        train_images = images[:50000]
        train_labels = labels[:50000]
        validation_images = images[50000:]
        validation_labels = labels[:50000]

        filter_1,filter_2,filter_3 = self.getParameters()
        feature_map_1 = self.convolution_1(train_images,filter_1)
        pooled_feature_map_1 = self.pooling_1(feature_map_1)
        feature_map_2 = self.convolution_2(pooled_feature_map_1,filter_2)
        pooled_feature_map_2 = self.pooling_2(feature_map_2)
        feature_map_3 = self.convolution_3(pooled_feature_map_2,filter_3)
        pooled_feature_map_3 = self.pooling_3(feature_map_3)
        
    def imageLoader(self):
        images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
        labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")    

        return images,labels

    def getParameters(self):
        """Loading parameters"""

        #============Filter: Convolution 1============#
        filter_1 = np.zeros((8,9))
        for i in range(8):
            filter_1[i,:] = np.arange(1,10)
        filter_1 = filter_1.reshape((8,3,3))

        #============Filter: Convolution 2============#
        filter_2 = np.zeros((16,9))
        for i in range(16):
            filter_2[i,:] = np.arange(1,10)
        filter_2 = filter_2.reshape((16,3,3))

        #============Filter: Convolution 3============#
        filter_3 = np.zeros((32,9))
        for i in range(32):
            filter_3[i,:] = np.arange(1,10)
        filter_3 = filter_3.reshape((32,3,3))

        return filter_1,filter_2,filter_3

    def convolution_1(self,train_images,filter):
        """Handles the first convolution. 
        Passes the 28x28 image through 8 3x3 filters with a stride of 1.
        Padded the image to create a 30x30 image so that the final output after convolution would be a 28x28 image.
        Takes the sum of the 3x3 filter after each scan.
        Returns a (8,28,28) feature map.
        """
        image = train_images[0]
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
    

model = convModel()
model.run()

