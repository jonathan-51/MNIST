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

        filter = self.getParameters()
        feature_map_1 = self.convolution_1(train_images,filter)
        pooled_feature_map_1 = self.pooling_1(feature_map_1)

    def imageLoader(self):
        images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
        labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")    

        return images,labels

    def getParameters(self):
        """Loading parameters"""

        #============Filter============#
        filter = np.zeros((8,9))
        for i in range(8):
            filter[i,:] = np.arange(1,10)
        filter = filter.reshape((8,3,3))
        return filter

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
model = convModel()
model.run()

