import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

labelDict = {'QG_ptD':'Jet energy sharing, p$_T$D','QG_mult':'Multiplicity','QG_axis2':'Jet minor axis','jet_pt':'Jet raw pT','jetPt':'Jet corrected pT','genJetPt':'Gen pT'}
binningDict = {'QG_ptD':np.arange(0.1,1.05,0.05),'QG_mult':np.arange(0.0,81.0,1.0),'QG_axis2':np.arange(0.0,0.15,0.005),'jet_pt':np.arange(0.0,1020.0,20.0),'jet_corr_pt':np.arange(0.0,1020.0,20.0),'genJetPt':np.arange(0.0,1020.0,20.0)}
yRangeDict = {'QG_ptD':[0.8,1.2],'QG_mult':[0.8,1.2],'QG_axis2':[0.8,1.2],'jet_pt':[0.8,1.2],'jet_corr_pt':[0.8,1.2],'genJetPt':[0.8,1.2]}

#Callback that produces training loss plots at the end of each epoch
class plotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.epoch = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.logs = []
        self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs = {}):
        self.logs.append(logs)
        self.x.append(self.epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.epoch += 1

        plt.plot(self.x[1:], self.losses[1:], label = 'Training loss')
        plt.plot(self.x[1:], self.val_losses[1:], label = 'Validation loss')
        plt.legend()
        plt.yscale('log')
        plt.title('Training and validation losses')
        plt.ylabel('Epoch loss')
        plt.xlabel('Epoch')

	plt.savefig('plots/Loss_plot.pdf')
	plt.clf()

#Callback that produces response plots from given test set at the end of each epoch
class plotResponses(Callback):
	def __init__(self, inputData, truthData, folder):
		self.epoch = 0
		self.inputs = inputData
		self.truths = truthData
		self.outputFolder = folder

	def on_epoch_end(self, epoch, logs={}):
		self.epoch += 1
		
		if self.epoch % 10 == 0:
			predictions = self.model.predict([self.inputs[0], self.inputs[1], self.inputs[2], self.inputs[3]])[:,0]
			DNNResponse = predictions*self.truths.jetPt/self.truths.genJetPt
			L1L2L3Response = self.truths.jetPt/self.truths.genJetPt
			
			#Distribution of predicted and L1L2L3 responses for UD and G jets
			#UD
			binning = np.arange(0.8, 1.3, 0.01)
            
			meanDNN = np.mean(DNNResponse[(self.truths.isPhysUDS==1)])
			stdDNN = np.std(DNNResponse[(self.truths.isPhysUDS==1)])
			meanL1L2L3 = np.mean(L1L2L3Response[(self.truths.isPhysUDS==1)])
			stdL1L2L3 = np.std(L1L2L3Response[(self.truths.isPhysUDS==1)])

			plt.hist(DNNResponse[self.truths.isPhysUDS==1], bins=binning, label='$\mu$: %0.3f, $\sigma$: %0.3f DNN'%(meanDNN, stdDNN), alpha=0.8)
			plt.hist(L1L2L3Response[self.truths.isPhysUDS==1], bins=binning, label='$\mu$: %0.3f, $\sigma$: %0.3f L1L2L3'%(meanL1L2L3, stdL1L2L3), alpha=0.8)
			plt.legend()
			plt.title('UD jet response distribution')
			plt.xlabel('Response')
			plt.ylabel('Jets')
			plt.yscale('log', nonposy='clip')
			plt.savefig(self.outputFolder+'/responseDistributionUD.pdf')
			plt.clf()

            #G
			binning = np.arange(0.8, 1.3, 0.01)

			meanDNN = np.mean(DNNResponse[self.truths.isPhysG==1])
			stdDNN = np.std(DNNResponse[self.truths.isPhysG==1])
			meanL1L2L3 = np.mean(L1L2L3Response[self.truths.isPhysG==1])
			stdL1L2L3 = np.std(L1L2L3Response[self.truths.isPhysG==1])

			plt.hist(DNNResponse[self.truths.isPhysG==1], bins=binning, label='$\mu$: %0.3f, $\sigma$: %0.3f DNN'%(meanDNN, stdDNN), alpha=0.8)
			plt.hist(L1L2L3Response[self.truths.isPhysG==1], bins=binning, label='$\mu$: %0.3f, $\sigma$: %0.3f L1L2L3'%(meanL1L2L3, stdL1L2L3), alpha=0.8)
 			plt.legend()
			plt.title('UD jet response distribution')
			plt.xlabel('Response')
			plt.ylabel('Jets')
			plt.yscale('log', nonposy='clip')
			plt.savefig(self.outputFolder+'/responseDistributionG.pdf')
			plt.clf()
			####End of response distributions
			
			#Mean and std.dev of responses w.r.t. to variables
			column=''
			toPlot=['genJetPt','QG_ptD','QG_mult', 'QG_axis2']
			for column in toPlot:
				print column
				binning = binningDict[column]
				midPoint = 1.0*(binning[1]-binning[0])/2.0
				xPoints = binning+midPoint
				UDJetIndices = (self.truths.isPhysUDS==1)
				GJetIndices = (self.truths.isPhysG==1)
				digitizedUD = np.digitize(self.truths[UDJetIndices][column],binning)
				digitizedG = np.digitize(self.truths[GJetIndices][column],binning)


				meansUD_DNN = np.zeros(len(binning))
				meansG_DNN = np.zeros(len(binning))
				meansUD_L1L2L3 = np.zeros(len(binning))
				meansG_L1L2L3 = np.zeros(len(binning))
				stdUD_DNN = np.zeros(len(binning))
				stdG_DNN = np.zeros(len(binning))
				stdUD_L1L2L3 = np.zeros(len(binning))
				stdG_L1L2L3 = np.zeros(len(binning))
				for i in range(1, len(binning)):
					meansUD_DNN[i-1] = np.mean(DNNResponse[UDJetIndices][digitizedUD==i])
					meansG_DNN[i-1] = np.mean(DNNResponse[GJetIndices][digitizedG==i])
					meansUD_L1L2L3[i-1] = np.mean(L1L2L3Response[UDJetIndices][digitizedUD==i])
					meansG_L1L2L3[i-1] = np.mean(L1L2L3Response[GJetIndices][digitizedG==i])
					stdUD_DNN[i-1] = np.std(DNNResponse[UDJetIndices][digitizedUD==i])
					stdG_DNN[i-1] = np.std(DNNResponse[GJetIndices][digitizedG==i])
					stdUD_L1L2L3[i-1] = np.std(L1L2L3Response[UDJetIndices][digitizedUD==i])
					stdG_L1L2L3[i-1] = np.std(L1L2L3Response[GJetIndices][digitizedG==i])

				#Clean out nans that happen for example if the bins are empty
				meansUD_DNN = np.nan_to_num(meansUD_DNN)
				meansG_DNN = np.nan_to_num(meansG_DNN)
				meansUD_L1L2L3 = np.nan_to_num(meansUD_L1L2L3)
				meansG_L1L2L3 = np.nan_to_num(meansG_L1L2L3)
				stdUD_DNN = np.nan_to_num(stdUD_DNN)
				stdG_DNN = np.nan_to_num(stdG_DNN)
				stdUD_L1L2L3 = np.nan_to_num(stdUD_L1L2L3)
				stdG_L1L2L3 = np.nan_to_num(stdG_L1L2L3)

				#UD##
				fig,(a0,a1) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
				a0.scatter(xPoints, meansUD_DNN,label='DNN', color='blue', s=8)
				a0.fill_between(xPoints, meansUD_DNN-stdUD_DNN,meansUD_DNN+stdUD_DNN, alpha=0.4, label='$\pm 1\sigma$',color='blue')
				a0.scatter(xPoints, meansUD_L1L2L3,label='L1L2L3', color='green', s=8)
				a0.fill_between(xPoints, meansUD_L1L2L3-stdUD_L1L2L3,meansUD_L1L2L3+stdUD_L1L2L3, alpha=0.4, label='$\pm 1\sigma$',color='green')
				a0.set_title('Mean UD responses and std w.r.t to '+labelDict[column])
				a0.set_ylim(yRangeDict[column][0],yRangeDict[column][1])
				a0.set_xlim(binning[0],binning[-1])
				a0.set_ylabel('Response')
				a0.legend()
				a0.plot([0, (binning[-1]+(binning[1]-binning[0]))], [1,1], 'k--')
				a0.set_xticks(binning[::4])

				#The distribution of jets in the lower plot
				n,bins,s = a1.hist(self.truths[UDJetIndices][column], bins=binning,
									weights=np.ones_like(self.truths[UDJetIndices][column]/float(np.sum(UDJetIndices))))
				a1.set_ylabel('Jet fraction')
				a1.set_xlabel(labelDict[column])
				plt.savefig(self.outputFolder+'/meanUDResponseVs'+column+'.pdf')
				plt.clf()


                binning = binningDict[column]
                midPoint = 1.0*(binning[1]-binning[0])/2.0
                xPoints = binning+midPoint
                #UD##
                fig,(a0,a1) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
                a0.scatter(xPoints, meansUD_DNN,label='DNN', color='blue', s=8)
                a0.fill_between(xPoints, meansUD_DNN-stdUD_DNN,meansUD_DNN+stdUD_DNN, alpha=0.4, label='$\pm 1\sigma$',color='blue')
                a0.scatter(xPoints, meansUD_L1L2L3,label='L1L2L3', color='green', s=8)
                a0.fill_between(xPoints, meansUD_L1L2L3-stdUD_L1L2L3,meansUD_L1L2L3+stdUD_L1L2L3, alpha=0.4, label='$\pm 1\sigma$',color='green')
                a0.set_title('Mean UD responses and std w.r.t to '+labelDict[column])
                a0.set_ylim(yRangeDict[column][0],yRangeDict[column][1])
                a0.set_xlim(binning[0],binning[-1])
                a0.set_ylabel('Response')
                a0.legend()
                a0.plot([0, (binning[-1]+(binning[1]-binning[0]))], [1,1], 'k--')
                a0.set_xticks(binning[::4])

                #The distribution of jets in the lower plot
                n,bins,s = a1.hist(self.truths[UDJetIndices][column], bins=binning,
                                    weights=np.ones_like(self.truths[UDJetIndices][column]/float(np.sum(UDJetIndices))))
                a1.set_ylabel('Jet fraction')
                a1.set_xlabel(labelDict[column])
                plt.savefig(self.outputFolder+'/meanUDResponseVs'+column+'.pdf')
                plt.clf()



def getStandardCallbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=15, min_lr=1e-7)
    checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss',
                                 verbose=0, save_best_only=False,
                                 mode='auto')
    loss_plot = plotLearning()

    monitoring = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True, write_grads=True) 

#    return [reduce_lr, checkpoint, loss_plot, monitoring]
    return [reduce_lr, checkpoint, loss_plot]

def makePlots(inputData, truthData, folder):
	response = plotResponses(inputData, truthData, folder)
	return [response]
