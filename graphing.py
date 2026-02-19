import os
import numpy
import matplotlib.pyplot as plt
import idx2numpy
import time
import pandas as pd
import seaborn as sns

#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)
#Suppresses scientific notion when printing out floating point numbers
numpy.set_printoptions(suppress=True)


class Validation:
    def __init__(self):
        pass

#========================================================================================================================
# TEST SPECIFIC PLOTTING
#========================================================================================================================
    """
    Reads csv files of both epoch summary and Calibration Curve,

    Figure 1 plots all 4 different graphs
    Graph 1 = Loss vs Epoch | Graph 2 = Accuracy vs Epoch | Graph 3 = Calibration Curve | Graph 4 = Normalized Gradients

    Figure 2 plots the confusion matrix.
    A confusion matrix every 10 epochs from [1,10,20,30,40,50] for a total of 6 Confusion Matrices.
    """
    # Learning rate: 0.01
    def Test_01(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""
        
        epoch_summary = pd.read_csv("Test_1.1_LR0-01/epoch_summary_T1.1.csv")
        CC = pd.read_csv("Test_1.1_LR0-01/CC_V_T1.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.01)
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_1.1_LR0-01/CM_T1.1/CM_e", file_name_2 = "_T1.1.npz",LR=0.01)

        plt.show()
        return
    
    # Learning rate: 0.001
    def Test_001(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_2.1_LR0-001/epoch_summary_T2.1.csv")
        CC = pd.read_csv("Test_2.1_LR0-001/CC_V_T2.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.001)
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_2.1_LR0-001/CM_T2.1/CM_e", file_name_2 = "_T2.1.npz",LR=0.001)

        plt.show()
        return
    
    # Learning rate: 0.005
    def Test_005(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_3.1_LR0-005/epoch_summary_T3.1.csv")
        CC = pd.read_csv("Test_3.1_LR0-005/CC_V_T3.1.csv")
        CC2 = pd.read_csv("Test_3.2_LR0-005/CC_V_T3.2.csv")
        fig1 = self.Plotting(epoch_summary,CC,LR=0.005)
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_3.1_LR0-005/CM_T3.1/CM_e", file_name_2 = "_T3.1.npz",LR=0.005)

        plt.show()
        return

    # Learning rate: 0.00005
    def Test_00005(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_4.1_LR0-00005/epoch_summary_T4.1.csv")
        CC = pd.read_csv("Test_4.1_LR0-00005/CC_V_T4.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.00005)
        fig2 = self.ConfusionMatrix(file_name_1 = "TTest_4.1_LR0-00005/CM_T4.1/CM_e", file_name_2 = "_T4.1.npz",LR=0.00005)

        plt.show()
        return
    
    # Learning rate: 0.0001
    def Test_0001(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_5.1_LR0-0001/epoch_summary_T5.1.csv")
        CC = pd.read_csv("Test_5.1_LR0-0001/CC_V_T5.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.0001) 
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_5.1_LR0-0001/CM_T5.1/CM_e", file_name_2 = "_T5.1.npz",LR=0.0001)

        plt.show()
    
    def Plotting(self,epoch_summary,CC,LR):
        "Handles Plotting all 4 graphs for figure 1"

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,5))   # Initializing a (2,2) figure

        plt.suptitle(f"Model's results from a LR of {LR}",size=30)
        plt.subplots_adjust(hspace=0.5)

        epoch = self.Epoch(epoch_summary)   # getting epoch values from specific Test

        #======================================================================================================
        # LOSS
        #======================================================================================================
        ax1 = self.Loss(epoch,epoch_summary,ax1)    # Loss Subplot
        ax1.set_title("Loss vs Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        #======================================================================================================
        # ACCURACY
        #======================================================================================================
        ax2 = self.Accuracy(epoch,epoch_summary,ax2)    # Accuracy Subplot
        ax2.set_title("Accuracy vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        #======================================================================================================
        # CALIBRATION CURVE
        #======================================================================================================
        ax3 = self.Calibration(CC,ax3)  # Calibration Curve Subplot
        ax3.set_title("Calibration Cruve")
        ax3.set_ylabel("Preidicted Probablity per Bin")
        ax3.set_xlabel("True Accuracy per Bin")
        ax3.grid(True)
        
        #======================================================================================================
        # NORMALIZED GRADIENT
        #======================================================================================================
        ax4 = self.NormGrad(epoch,epoch_summary,ax4)    # Normalized Gradient Subplot
        ax4.set_title("Normalized Gradient vs Epoch")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Normalized Gradient")
        ax4.grid(True)

        return fig

    def Epoch(self,epoch_summary):
        """Returns the number of epochs ran by a specific test"""

        epoch = epoch_summary['Epoch']

        return epoch

    def Loss(self,epoch,epoch_summary,ax):
        """Returns the subplot for both training and validation loss"""

        train_loss = epoch_summary['Train Loss']
        val_loss = epoch_summary['Val Loss']

        ax.plot(epoch,train_loss,label="Training",color="Blue")
        ax.plot(epoch,val_loss,label="Validation",color="Green")

        return ax

    def Accuracy(self,epoch,epoch_summary,ax):
        """Returns the training and validation accuracy"""

        train_acc = epoch_summary['Train Acc']
        train_acc = train_acc.astype(float)

        val_acc = epoch_summary['Val Acc']
        val_acc = val_acc.astype(float)   

        ax.plot(epoch,train_acc,label="Training",color="Blue")
        ax.plot(epoch,val_acc,label="Validation",color="Green")

        return ax
    
    def ConfusionMatrix(self,file_name_1,file_name_2,LR):
        """Returns the figure for Confusion Matrix."""

        fig1, axes = plt.subplots(2,3,figsize=(8,8))    # Initializing (2,3) figure
        fig1.suptitle(f"Confusion Matrix every 10 Epochs with a LR of {LR}")
        fig1.subplots_adjust(hspace=0.5)
        
        # Storing each subplot in a dictionary
        axes_dict = {"ax1":axes[0,0],
                     "ax2":axes[0,1],
                     "ax3":axes[0,2],
                     "ax4":axes[1,0],
                     "ax5":axes[1,1],
                     "ax6":axes[1,2],}
        
        epoch = numpy.array([1,10,20,30,40,50]) # Epoch Numbers

        #====================================================================================
        # EPOCH #1
        #====================================================================================
        data = numpy.load(f"{file_name_1}1{file_name_2}")
        correct_matrix = data['confusion_matrx']

        incorrect_matrix = correct_matrix.copy()

        mask= numpy.eye(len(correct_matrix),dtype=bool) # Boolean Identity Matrix

        correct_matrix = numpy.where(mask,correct_matrix,numpy.nan) # Replaces all false values (incorrect predictions) with nan
        
        initial_correct_max = numpy.nanmax(correct_matrix)  # getting max value after replacing incorrect predictions with nan
        correct_matrix = correct_matrix/initial_correct_max # Normalizing all values between 0-1


        numpy.fill_diagonal(incorrect_matrix,numpy.nan) # Replaces all true values (correct predictions) with nan
        initial_incorrect_max = numpy.nanmax(incorrect_matrix)  # max value after replacing correct predictions with nan
        incorrect_matrix = incorrect_matrix/initial_incorrect_max   # Normalizing all values between 0-1

        # Heatmap for correct predictions only
        sns.heatmap(correct_matrix,cbar=False, annot=False, cmap='Greys', vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax1'], annot_kws={'size': 10})

        #Heatmap for incorrect predictions only
        sns.heatmap(incorrect_matrix,cbar=False, annot=False, cmap='Oranges',vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax1'], annot_kws={'size': 10})

        # Universal color bar for both heatmaps
        cbar = fig1.colorbar(axes[0,0].collections[0], ax = axes, orientation='vertical',fraction=0.02,pad=0.03)
        cbar = fig1.colorbar(axes[0,0].collections[1], ax = axes, orientation='vertical',fraction=0.02,pad=0.03)

        axes_dict[f'ax1'].set_title(f"Epoch 1")
        axes_dict[f'ax1'].set_xlabel("Predicted Label")
        axes_dict[f'ax1'].set_ylabel("True Label")
        axes_dict[f'ax1'].set_xticks(numpy.arange(10)+0.5)
        axes_dict[f'ax1'].set_xticklabels(numpy.arange(10))

        #====================================================================================
        # EPOCH #10,20,30,40,50
        #====================================================================================
        for subplot_num in range (2,7):
            data = numpy.load(f"{file_name_1}{epoch[subplot_num-1]}{file_name_2}")
            correct_matrix = data['confusion_matrx']

            
            incorrect_matrix = correct_matrix.copy()

            mask= numpy.eye(len(correct_matrix),dtype=bool) # Boolean Identity Matrix

            correct_matrix = numpy.where(mask,correct_matrix,numpy.nan) # Replaces all false values (incorrect predictions) with nan
            
            correct_matrix = correct_matrix/initial_correct_max # Normalizing all values between 0-1, SCALED BY FIRST EPOCH (used as reference)

            numpy.fill_diagonal(incorrect_matrix,numpy.nan) # Replaces all true values (correct predictions) with nan
            incorrect_matrix = incorrect_matrix/initial_incorrect_max   # Normalizing all values between 0-1, SCALED BY FIRST EPOCH (used as reference)

            # Heatmap for correct predictions only
            sns.heatmap(correct_matrix,cbar=False, annot=False, cmap='Greys', vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax{subplot_num}'], annot_kws={'size': 10})
            # Heatmap for icorrect predictions only
            sns.heatmap(incorrect_matrix,cbar=False, annot=False, cmap='Oranges',vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax{subplot_num}'], annot_kws={'size': 10})

            

            axes_dict[f'ax{subplot_num}'].set_title(f"Epoch {epoch[subplot_num-1]}")
            axes_dict[f'ax{subplot_num}'].set_xlabel("Predicted Label")
            axes_dict[f'ax{subplot_num}'].set_ylabel("True Label")
            axes_dict[f'ax{subplot_num}'].set_xticks(numpy.arange(10)+0.5)
            axes_dict[f'ax{subplot_num}'].set_xticklabels(numpy.arange(10))

        
        return fig1
    
    def Calibration(self,CC,ax):
        "Returns subplot for calibration curve"
        val_pp = pd.Series(numpy.zeros(10))
        val_acc_bin = pd.Series(numpy.zeros(10))

        #====================================================================================
        # PREDICTED PROBABILITY
        #====================================================================================
        #Bins 0 - 8
        for bin in range(1,9):
            val_pp[bin] = (CC[f'PP0.{bin}-0.{bin+1}'].sum())/len(CC)    # Calculating average predicted probability for each bin between 0-8
        #Bin 9
        val_pp[9] = (CC['PP0.9-1.0'].sum())/len(CC) # Calculating average predicted probability for bin 9

        #====================================================================================
        # ACCURACY
        #====================================================================================
        #Bins 0 - 8
        for bin in range(1,9):
            val_acc_bin[bin] = (CC[f'A0.{bin}-0.{bin+1}'].sum())/len(CC)    # Calculating average predicted probability for each bin between 0-8
        #Bin 9
        val_acc_bin[9] = (CC['A0.9-1.0'].sum())/len(CC) # Calculating average predicted probability for bin 9
        
        # Computing diagonal line
        x = numpy.linspace(0,100,11)
        y = x

        ax.plot(val_acc_bin*100,val_pp*100,color="brown")
        ax.plot(x,y,color="Black")

        return ax
    
    def NormGrad(self,epoch,epoch_summary,ax):
        """Returns subplot for normalized gradient"""
        norm_grad = epoch_summary['Grad Norm']

        ax.plot(epoch,norm_grad,color="red")    
        return ax

class betweenTest:
    def __init__(self):
        pass
    
    def plotting_compare(self):

        tests = {"test_0_01" : pd.read_csv("Test_1.1_LR0-01/epoch_summary_T1.1.csv"),
                 "test_0_001" : pd.read_csv("Test_2.1_LR0-001/epoch_summary_T2.1.csv"),
                 "test_0_005" : pd.read_csv("Test_3.1_LR0-005/epoch_summary_T3.1.csv"),
                 "test_0_00005" : pd.read_csv("Test_4.1_LR0-00005/epoch_summary_T4.1.csv"),
                 "test_0_0001" : pd.read_csv("Test_5.1_LR0-0001/epoch_summary_T5.1.csv")}
        
        tests_calibration = {"test_0_01" : pd.read_csv("Test_1.1_LR0-01/CC_V_T1.1.csv"),
                             "test_0_001" : pd.read_csv("Test_2.1_LR0-001/CC_V_T2.1.csv"),
                             "test_0_005" : pd.read_csv("Test_3.1_LR0-005/CC_V_T3.1.csv"),
                             "test_0_00005" : pd.read_csv("Test_4.1_LR0-00005/CC_V_T4.1.csv"),
                             "test_0_0001" : pd.read_csv("Test_5.1_LR0-0001/CC_V_T5.1.csv")}

        fig1, axes1 = plt.subplots(2,2,figsize=(10,5))
        fig1 = plt.suptitle("Loss comparison across 5 different learning rates",size=20)
        fig1 = plt.subplots_adjust(hspace=0.3)

        fig2, axes2 = plt.subplots(2,2,figsize=(10,5))
        fig2 = plt.suptitle("Accuracy comparison across 5 different learning rates",size=20)
        fig2 = plt.subplots_adjust(hspace=0.3)    
        epoch = pd.Series(numpy.arange(1,51,1))

        axes1 = self.loss_compare(axes1,epoch,tests)
        axes2 = self.accuracy_compare(axes2,epoch,tests)
        

        fig3,axes3 = plt.subplots()
        fig2 = plt.suptitle("Calibration Curve across 5 different learning rates",size=20)
        axes3 = self.calibration_compare(axes3,epoch,tests_calibration)
        
        plt.show()
        return

    #Handles Loss across all 5 tests
    def loss_compare(self,axes1,epoch,tests):
        test_configs = [("test_0_01","0.01","green"),
                        ("test_0_001","0.001","red"),
                        ("test_0_005","0.005","orange"),
                        ("test_0_00005","0.00005","blue"),
                        ("test_0_0001","0.0001","black")]
        for j in range(0,2):
            for test,lr,color in test_configs:
                if j == 0:
                    self.loss_training(axes1[0,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)
                else:
                    self.loss_val(axes1[0,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)

        for j in range(0,2):
            for test,lr,color in test_configs[:3]:  
                if j == 0:
                    self.loss_training(axes1[1,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)
                else:
                    self.loss_val(axes1[1,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)

        axes1[0,0].set_title("Training Loss vs Epoch") 
        axes1[0,1].set_title("Validation Loss vs Epoch")
        axes1[1,0].set_title("Training Loss vs Epoch")
        axes1[1,1].set_title("Validation Loss vs Epoch")

        for i in range(0,2):
            for j in range(0,2):
                axes1[i,j].set_xlabel("Epoch")
                axes1[i,j].set_ylabel("Loss")
                axes1[i,j].legend()
                axes1[i,j].grid(True)
        return axes1
    
    def loss_training(self,ax,epoch,epoch_summary,label_test,color_test):

        train_loss = epoch_summary["Train Loss"]
        ax.plot(epoch,train_loss,label = label_test,color = color_test)
        
        return ax

    def loss_val(self,ax,epoch,epoch_summary,label_test,color_test):

        val_loss = epoch_summary["Val Loss"]
        ax.plot(epoch,val_loss,label = label_test,color = color_test,linestyle ='-' )

        return ax
    
    #Handles Acuracy  across all 5 tests
    def accuracy_compare(self,axes2,epoch,tests):

        test_configs = [("test_0_01","0.01","green"),
                        ("test_0_001","0.001","red"),
                        ("test_0_005","0.005","orange"),
                        ("test_0_00005","0.00005","blue"),
                        ("test_0_0001","0.0001","black")]
        for j in range(0,2):
            for test,lr,color in test_configs:
                if j == 0:
                    self.accuracy_training(axes2[0,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)
                else:
                    self.accuracy_val(axes2[0,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)

        for j in range(0,2):
            for test,lr,color in test_configs[:3]:  
                if j == 0:
                    self.accuracy_training(axes2[1,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)
                else:
                    self.accuracy_val(axes2[1,j],epoch,tests[f"{test}"],label_test=lr,color_test = color)


        axes2[0,0].set_title("Training Accuracy vs Epoch") 
        axes2[0,1].set_title("Validation Accuracy vs Epoch")
        axes2[1,0].set_title("Training Accuracy vs Epoch")
        axes2[1,1].set_title("Validation Accuracy vs Epoch")

        for i in range(0,2):
            for j in range(0,2):
                axes2[i,j].set_xlabel("Epoch")
                axes2[i,j].set_ylabel("Accuracy")
                axes2[i,j].legend()
                axes2[i,j].grid(True)
        return axes2

    def accuracy_training(self,ax,epoch,epoch_summary,label_test,color_test):
        
        train_accuracy = epoch_summary["Train Acc"]
        ax.plot(epoch,train_accuracy,label=label_test,color=color_test)

        return ax
    
    def accuracy_val(self,ax,epoch,epoch_summary,label_test,color_test):
    
        val_accuracy = epoch_summary["Val Acc"]
        ax.plot(epoch,val_accuracy,label=label_test,color=color_test)

        return ax

    def calibration_compare(self,axes3,epoch,tests_calibration):
        test_configs = [("test_0_01","0.01","green"),
                        ("test_0_001","0.001","red"),
                        ("test_0_005","0.005","orange"),
                        ("test_0_00005","0.00005","blue"),
                        ("test_0_0001","0.0001","black")]
        
        for test, lr, color in test_configs:

            val_pp = pd.Series(numpy.zeros(10))
            val_acc_bin = pd.Series(numpy.zeros(10))

                #Bins 0 - 8
            for bin in range(1,9):
                val_pp[bin] = ((tests_calibration[f"{test}"])[f'PP0.{bin}-0.{bin+1}'].sum())/len((tests_calibration[f"{test}"]))    # Calculating average predicted probability for each bin between 0-8
                #Bin 9
                val_pp[9] = ((tests_calibration[f"{test}"])['PP0.9-1.0'].sum())/len((tests_calibration[f"{test}"])) # Calculating average predicted probability for bin 9
            for bin in range(1,9):
                val_acc_bin[bin] = ((tests_calibration[f"{test}"])[f'A0.{bin}-0.{bin+1}'].sum())/len((tests_calibration[f"{test}"]))    # Calculating average predicted probability for each bin between 0-8
                #Bin 9
                val_acc_bin[9] = ((tests_calibration[f"{test}"])['A0.9-1.0'].sum())/len((tests_calibration[f"{test}"])) # Calculating average predicted probability for bin 9
                
                
            axes3.plot(val_acc_bin*100,val_pp*100,label = lr, color = color)

        # Computing diagonal line
        x = numpy.linspace(0,100,11)
        y = x
        axes3.plot(x,y,label="Control",color="Brown")

        axes3.set_ylabel("Preidicted Probablity per Bin")
        axes3.set_xlabel("True Accuracy per Bin")
        axes3.legend()
        axes3.grid(True)

        return axes3

    def calibration_lr_0_005(self):
        CC = pd.read_csv("Test_3.1_LR0-005/CC_V_T3.1.csv")
        CC_TS = pd.read_csv("Test_3.2_LR0-005/CC_V_T3.2.csv")
        
        calibration_dict ={"CC":CC.iloc[16],
                      "CC_TS":CC_TS}

        test_configs = [("CC","Original","green"),
                        ("CC_TS","Temperature Scaled","red"),]
        
        fig, ax = plt.subplots(1,figsize=(10,5))

        for calibration,types,color in test_configs:
            val_pp = pd.Series(numpy.zeros(10))
            val_acc_bin = pd.Series(numpy.zeros(10))

                #Bins 0 - 8
            for bin in range(1,9):
                val_pp[bin] = ((calibration_dict[f"{calibration}"])[f'PP0.{bin}-0.{bin+1}'])    # Calculating average predicted probability for each bin between 0-8
                #Bin 9
                val_pp[9] = ((calibration_dict[f"{calibration}"])['PP0.9-1.0']) # Calculating average predicted probability for bin 9
            for bin in range(1,9):
                val_acc_bin[bin] = ((calibration_dict[f"{calibration}"])[f'A0.{bin}-0.{bin+1}'])   # Calculating average predicted probability for each bin between 0-8
                #Bin 9
                val_acc_bin[9] = ((calibration_dict[f"{calibration}"])['A0.9-1.0']) # Calculating average predicted probability for bin 9
                
                
            ax.plot(val_acc_bin*100,val_pp*100,label = types, color = color)

        # Computing diagonal line
        x = numpy.linspace(0,100,11)
        y = x
        ax.plot(x,y,label="Control",color="Brown")

        ax.set_ylabel("Preidicted Probablity per Bin")
        ax.set_xlabel("True Accuracy per Bin")
        ax.legend()
        ax.grid(True)    

        plt.show()
        return

class Test:
    def __init__(self):
        pass
    def Plot(self):
        test_summary = pd.read_csv("Test_Summary.csv")
        epoch = test_summary["Epoch"]
        loss = test_summary["Loss"]
        accuracy = test_summary["Accuracy"]
        ECE = test_summary["ECE"]

        fig1, axes = plt.subplots(1,3,figsize=(10,5))
        plt.suptitle("Test Summary Across All LR0.005 Parameters")
        axes[0] = self.PlotLoss(axes[0],epoch,loss)
        axes[1] = self.PlotAccuracy(axes[1],epoch,accuracy)
        axes[2] = self.PlotECE(axes[2],epoch,ECE)
        fig2 = self.ConfusionMatrixHeatMap()
        fig3 = self.ConfusionMatrix()
        #axes[3] = self.PlotComparison(axes[3],epoch,loss,accuracy,ECE)

        plt.show()

        return
    
    def PlotLoss(self,ax,epoch,loss):
        ax.plot(epoch,loss)
        ax.set_title("Loss vs Each Epoch's Parameters")
        ax.set_xlabel("Epoch Parameters")
        ax.set_ylabel("Loss")
        return ax
    
    def PlotAccuracy(self,ax,epoch,accuracy):
        ax.plot(epoch,accuracy)
        ax.set_title("Accuracy vs Each Epoch's Parameters")
        ax.set_xlabel("Epoch Parameters")
        ax.set_ylabel("Accuracy")
        return ax
    
    def PlotECE(self,ax,epoch,ECE):
        ax.plot(epoch,ECE)
        ax.set_title("ECE vs Each Epoch's Parameters")
        ax.set_xlabel("Epoch Parameters")
        ax.set_ylabel("ECE")
        return ax
    
    def PlotComparison(self,ax,epoch,loss,accuracy,ECE):
        loss_normalized = loss/loss.max()
        accuracy_normalized = accuracy/accuracy.max()
        ECE_normalized = ECE/ECE.max()

        ax.plot(epoch,loss_normalized,label="Loss",color="red")
        ax.plot(epoch,accuracy_normalized,label="Accuracy",color="green")
        ax.plot(epoch,ECE_normalized,label="ECE",color="blue")

        ax.set_title("Loss/Accuracy/ECE vs Each Epoch's Parameters")
        ax.set_xlabel("Epoch Parameters")
        ax.set_ylabel("Normalized Between 0 and 1")
        ax.legend()

        return ax

    def ConfusionMatrixHeatMap(self):
        """Returns the figure for Confusion Matrix."""

        fig, ax1 = plt.subplots(1,figsize=(8,8))    # Initializing (2,3) figure
        fig.suptitle(f"Confusion Matrix")
        fig.subplots_adjust(hspace=0.5)
    
        #====================================================================================
        # EPOCH #1
        #====================================================================================
        data = numpy.load(f"Test_3.1_LR0-005/CM_T3.1/CM_e15_T3.1.npz")
        correct_matrix = data['confusion_matrx']

        incorrect_matrix = correct_matrix.copy()

        mask= numpy.eye(len(correct_matrix),dtype=bool) # Boolean Identity Matrix

        correct_matrix = numpy.where(mask,correct_matrix,numpy.nan) # Replaces all false values (incorrect predictions) with nan
        
        initial_correct_max = numpy.nanmax(correct_matrix)  # getting max value after replacing incorrect predictions with nan
        correct_matrix = correct_matrix/initial_correct_max # Normalizing all values between 0-1


        numpy.fill_diagonal(incorrect_matrix,numpy.nan) # Replaces all true values (correct predictions) with nan
        initial_incorrect_max = numpy.nanmax(incorrect_matrix)  # max value after replacing correct predictions with nan
        incorrect_matrix = incorrect_matrix/initial_incorrect_max   # Normalizing all values between 0-1

        # Heatmap for correct predictions only
        sns.heatmap(correct_matrix,cbar=False, annot=True, cmap='Greys', vmin=0,vmax=1, fmt='.1f', ax = ax1, annot_kws={'size': 10})

        #Heatmap for incorrect predictions only
        sns.heatmap(incorrect_matrix,cbar=False, annot=True, cmap='Oranges',vmin=0,vmax=1, fmt='.1f', ax = ax1, annot_kws={'size': 10})

        # Universal color bar for both heatmaps
        cbar = fig.colorbar(ax1.collections[0], ax = ax1, orientation='vertical',fraction=0.02,pad=0.03)
        cbar = fig.colorbar(ax1.collections[1], ax = ax1, orientation='vertical',fraction=0.02,pad=0.03)

        ax1.set_title(f"Epoch 1")
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")
        ax1.set_xticks(numpy.arange(10)+0.5)
        ax1.set_xticklabels(numpy.arange(10))

        return fig

    def ConfusionMatrix(self):
        """Returns the figure for Confusion Matrix."""

        fig, ax1 = plt.subplots(1,figsize=(8,8))    # Initializing (2,3) figure
        fig.suptitle(f"Confusion Matrix")
        fig.subplots_adjust(hspace=0.5)
    
        #====================================================================================
        # EPOCH #1
        #====================================================================================
        data = numpy.load(f"Test_3.1_LR0-005/CM_T3.1/CM_e15_T3.1.npz")
        matrix = data['confusion_matrx']
        matrix_row_sum = numpy.sum(matrix,axis=1)
        matrix_percentage = (matrix/matrix_row_sum)*100
        # Heatmap for correct predictions only
        sns.heatmap(matrix_percentage,cbar=False,cmap='Blues', annot=True, fmt='.2f', ax = ax1, annot_kws={'size': 10})

        ax1.set_title(f"Epoch 1")
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")
        ax1.set_xticks(numpy.arange(10)+0.5)
        ax1.set_xticklabels(numpy.arange(10))
        ax1.grid(True)
        return fig
        
#Test1(epoch_summary_T1,CC_V_T1)
#Test2(epoch_summary_T2,CC_V_T2)
# Loss_Compare(epoch_summary_T1_1, epoch_summary_T2_1,epoch_summary_T3_1)
#lr_finder()

# plot = Validation()
# plot_compare = betweenTest()
# plot.Test_005()
# plot.Test_001()
# plot.Test_01()
# plot.Test_0001()
# plot_compare.calibration_lr_0_005()
# plot_compare.plotting_compare()

# test = Test()
# test.Plot()

class Report:
    def __init__(self):
        os.makedirs("Statistics", exist_ok=True)
        self.tests = [
            {"lr": 0.01,    "folder": "Test_1.1_LR0-01",    "epoch_csv": "epoch_summary_T1.1.csv", "cc_csv": "CC_V_T1.1.csv", "cm_folder": "CM_T1.1"},
            {"lr": 0.001,   "folder": "Test_2.1_LR0-001",   "epoch_csv": "epoch_summary_T2.1.csv", "cc_csv": "CC_V_T2.1.csv", "cm_folder": "CM_T2.1"},
            {"lr": 0.005,   "folder": "Test_3.1_LR0-005",   "epoch_csv": "epoch_summary_T3.1.csv", "cc_csv": "CC_V_T3.1.csv", "cm_folder": "CM_T3.1"},
            {"lr": 0.00005, "folder": "Test_4.1_LR0-00005", "epoch_csv": "epoch_summary_T4.1.csv", "cc_csv": "CC_V_T4.1.csv", "cm_folder": "CM_T4.1"},
            {"lr": 0.0001,  "folder": "Test_5.1_LR0-0001",  "epoch_csv": "epoch_summary_T5.1.csv", "cc_csv": "CC_V_T5.1.csv", "cm_folder": "CM_T5.1"},
        ]

    def run(self):
        self.training_validation_per_test()
        self.calibration_per_test()
        self.confusion_matrix_per_test()
        self.cross_lr_comparison()
        self.test_results()
        print("All plots saved to Statistics/")

    def training_validation_per_test(self):
        """Per learning rate: loss, accuracy, and gradient norm over epochs."""
        for test in self.tests:
            path = f"{test['folder']}/{test['epoch_csv']}"
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            epoch = df['Epoch']

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Training & Validation — LR {test['lr']}", size=14)

            axes[0].plot(epoch, df['Train Loss'], label='Train', color='blue')
            axes[0].plot(epoch, df['Val Loss'],   label='Val',   color='green')
            axes[0].set_title("Loss vs Epoch")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(epoch, df['Train Acc'], label='Train', color='blue')
            axes[1].plot(epoch, df['Val Acc'],   label='Val',   color='green')
            axes[1].set_title("Accuracy vs Epoch")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy (%)")
            axes[1].legend()
            axes[1].grid(True)

            axes[2].plot(epoch, df['Grad Norm'], color='red')
            axes[2].set_title("Gradient Norm vs Epoch")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Gradient Norm")
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig(f"Statistics/TrainVal_LR{test['lr']}.png", dpi=150, bbox_inches='tight')
            plt.close()

    def calibration_per_test(self):
        """Per learning rate: calibration curve averaged across all epochs."""
        for test in self.tests:
            path = f"{test['folder']}/{test['cc_csv']}"
            if not os.path.exists(path):
                continue
            CC = pd.read_csv(path)

            val_pp      = pd.Series(numpy.zeros(10))
            val_acc_bin = pd.Series(numpy.zeros(10))

            for b in range(1, 9):
                val_pp[b]      = CC[f'PP0.{b}-0.{b+1}'].mean()
                val_acc_bin[b] = CC[f'A0.{b}-0.{b+1}'].mean()
            val_pp[9]      = CC['PP0.9-1.0'].mean()
            val_acc_bin[9] = CC['A0.9-1.0'].mean()

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.suptitle(f"Calibration Curve (avg across epochs) — LR {test['lr']}", size=12)
            x = numpy.linspace(0, 100, 11)
            ax.plot(val_acc_bin * 100, val_pp * 100, color='brown', label='Model')
            ax.plot(x, x, color='black', linestyle='--', label='Perfect calibration')
            ax.set_xlabel("True Accuracy per Bin (%)")
            ax.set_ylabel("Predicted Probability per Bin (%)")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.savefig(f"Statistics/Calibration_LR{test['lr']}.png", dpi=150, bbox_inches='tight')
            plt.close()

    def confusion_matrix_per_test(self):
        """Per learning rate: confusion matrix at the final saved epoch, as percentages."""
        for test in self.tests:
            cm_dir = f"{test['folder']}/{test['cm_folder']}"
            if not os.path.exists(cm_dir):
                continue
            cm_files = sorted(
                [f for f in os.listdir(cm_dir) if f.endswith('.npz')],
                key=lambda f: int(''.join(filter(str.isdigit, f)))
            )
            if not cm_files:
                continue

            data   = numpy.load(f"{cm_dir}/{cm_files[-1]}")
            matrix = data['confusion_matrx']
            matrix_pct = (matrix / matrix.sum(axis=1, keepdims=True)) * 100

            fig, ax = plt.subplots(figsize=(7, 6))
            fig.suptitle(f"Confusion Matrix ({cm_files[-1]}) — LR {test['lr']}", size=12)
            sns.heatmap(matrix_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                        cbar_kws={'label': '% of True Label'}, annot_kws={'size': 8})
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            plt.tight_layout()
            plt.savefig(f"Statistics/ConfusionMatrix_LR{test['lr']}.png", dpi=150, bbox_inches='tight')
            plt.close()

    def cross_lr_comparison(self):
        """All 5 learning rates on one figure: train/val loss and accuracy."""
        colors = ['green', 'red', 'orange', 'blue', 'black']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training & Validation Comparison Across Learning Rates", size=14)

        for test, color in zip(self.tests, colors):
            path = f"{test['folder']}/{test['epoch_csv']}"
            if not os.path.exists(path):
                continue
            df    = pd.read_csv(path)
            epoch = df['Epoch']
            lr    = str(test['lr'])

            axes[0, 0].plot(epoch, df['Train Loss'], label=lr, color=color)
            axes[0, 1].plot(epoch, df['Val Loss'],   label=lr, color=color)
            axes[1, 0].plot(epoch, df['Train Acc'],  label=lr, color=color)
            axes[1, 1].plot(epoch, df['Val Acc'],    label=lr, color=color)

        for ax, (title, ylabel) in zip(axes.flat, [
            ("Training Loss",     "Loss"),
            ("Validation Loss",   "Loss"),
            ("Training Accuracy", "Accuracy (%)"),
            ("Validation Accuracy","Accuracy (%)"),
        ]):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig("Statistics/Comparison_All_LR.png", dpi=150, bbox_inches='tight')
        plt.close()

    def test_results(self):
        """Test set performance (loss, accuracy, ECE) across all 50 saved epoch parameters."""
        path = "Test_Summary.csv"
        if not os.path.exists(path):
            return
        df    = pd.read_csv(path)
        epoch = df['Epoch']

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle("Test Set Results Across Saved Epoch Parameters (LR 0.005)", size=13)

        axes[0].plot(epoch, df['Loss'],     color='blue')
        axes[0].set_title("Test Loss")
        axes[0].set_xlabel("Epoch Parameters")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        axes[1].plot(epoch, df['Accuracy'], color='green')
        axes[1].set_title("Test Accuracy")
        axes[1].set_xlabel("Epoch Parameters")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(True)

        axes[2].plot(epoch, df['ECE'],      color='red')
        axes[2].set_title("ECE (Expected Calibration Error)")
        axes[2].set_xlabel("Epoch Parameters")
        axes[2].set_ylabel("ECE")
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig("Statistics/Test_Results.png", dpi=150, bbox_inches='tight')
        plt.close()

report = Report()
report.run()
