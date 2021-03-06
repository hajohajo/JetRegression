# Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False
# /////////////////////

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd
import numpy as np
import glob, re, joblib
import hickle as hkl

from root_pandas import read_root
from keras.models import Model, load_model
from scipy.stats import sem

model = load_model('checkpoint_model.h5')
test_data_ = read_root("test_data.root")

# Read variables saved at training
n_particles, Ccand_variables, Ncand_variables, Pcand_variables, Global_variables, Gen_variables, Flavor_variables, Training_variables = hkl.load('Variables.hkl')

# Use same scaler with the weights from the training dataset
scaler = joblib.load("scaler.pkl")
test_inp = pd.DataFrame(scaler.transform(test_data_[Training_variables].values), columns=Training_variables)
test_trg = test_data_['target']

# Separate globals, charged, neutral and photon candidates to their own inputs
test_Ccands = test_inp[Ccand_variables]
test_Ncands = test_inp[Ncand_variables]
test_Pcands = test_inp[Pcand_variables]
test_Globals = test_inp[Global_variables]


# Reshaping pfCand arrays to fit LSTMs
test_Ccands = np.reshape(np.array(test_Ccands), (test_Ccands.shape[0], n_particles, test_Ccands.shape[1]/n_particles))
test_Ncands = np.reshape(np.array(test_Ncands), (test_Ncands.shape[0], n_particles, test_Ncands.shape[1]/n_particles))
test_Pcands = np.reshape(np.array(test_Pcands), (test_Pcands.shape[0], n_particles, test_Pcands.shape[1]/n_particles))

test_data_['predictions'] = model.predict([test_Ccands, test_Ncands, test_Pcands, test_Globals])
print test_data_['predictions'][:5]
print test_data_['target'][:5]


# Assuming learning target was the correction factor
test_data_['Response'] = test_data_['jetPt']/test_data_['genJetPt']
test_data_['jetPt_DNN'] = test_data_['jetPt']*test_data_['predictions']
test_data_['Response_DNN'] = test_data_['jetPt_DNN']/test_data_['genJetPt']

# List of pairs of variables to plots, comparing the jet corr pT and the DNN corrected pT
to_plot = [('genJetPt', 'Response'),
           ('QG_ptD', 'Response'),
           ('QG_axis2', 'Response'),
           ('QG_mult', 'Response')]

to_histogram = ['Response']

# Dictionary for binning the different variables
bin_dict = {'QG_ptD':np.arange(0.15, 1.00, 0.05),
            'QG_axis2':np.arange(0.0, 0.15, 0.005),
            'QG_mult':np.arange(1.0, 81.0, 1.0),
            'genJetPt':np.arange(80.0, 1020.0, 20.0),
	    'Response':np.arange(0.80, 1.21, 0.01)}
yrange_dict = {'Response':[0.8, 1.2]}

for type in ['isPhysUDS', 'isPhysG']:
    test_data=test_data_[test_data_[type]==1]
    for variables in to_plot:
        binning = bin_dict[variables[0]]
        step = binning[1]-binning[0]
        x = binning + step/2.0
        indices = np.digitize(test_data[variables[0]], binning)

        means_l1l2l3 = np.zeros(len(binning))
        means_DNN = np.zeros(len(binning))
        errors_l1l2l3 = np.zeros(len(binning))
        errors_DNN = np.zeros(len(binning))

        for i in range(1, len(binning)+1):
            means_l1l2l3[i-1] = np.mean(test_data[variables[1]][indices == i])
            means_DNN[i-1] = np.mean(test_data[variables[1]+'_DNN'][indices == i])
            errors_l1l2l3[i-1] = np.std(test_data[variables[1]][indices == i])
            errors_DNN[i-1] = np.std(test_data[variables[1]][indices == i])

        # Take care of possible nans, due to empty bins
        means_l1l2l3 = np.nan_to_num(means_l1l2l3)
        means_DNN = np.nan_to_num(means_DNN)
        errors_l1l2l3 = np.nan_to_num(errors_l1l2l3)
        errors_DNN = np.nan_to_num(errors_DNN)


        plt.scatter(x, means_l1l2l3, label='L1L2L3', color='blue', s=2)
        plt.fill_between(x, means_l1l2l3-errors_l1l2l3, means_l1l2l3+errors_l1l2l3,
                         alpha=0.4, label='$\pm 1\sigma$', color='blue')
        plt.scatter(x, means_DNN, label='DNN', color='orange', s=2)
        plt.fill_between(x, means_DNN-errors_DNN, means_DNN+errors_DNN,
                         alpha=0.4, label='$\pm 1\sigma$', color='orange')

        plt.legend()
        plt.plot([0, (binning[-1]+step)], [1, 1], 'k--')
        plt.ylim(yrange_dict[variables[1]][0], yrange_dict[variables[1]][1])
        plt.xlim(binning[0], binning[-1])
        plt.xticks(binning[::4])
        plt.ylabel(variables[1])
        plt.xlabel(variables[0])

        plt.savefig('plots/'+variables[1]+'_vs_'+variables[0]+'_'+type+'.pdf')
        plt.clf()

    for variable in to_histogram:
        mean_l1l2l3 = np.mean(test_data[variable])
        mean_DNN = np.mean(test_data[variable+'_DNN'])
        std_l1l2l3 = np.std(test_data[variable])
        std_DNN = np.std(test_data[variable+'_DNN'])

        plt.hist(test_data[variable], bins=bin_dict[variable], alpha=0.8, edgecolor='black', linewidth=1.2,
                 label='$\mu$: %0.3f, $\sigma$: %0.3f Regression' % (mean_l1l2l3, std_l1l2l3))
        plt.hist(test_data[variable+'_DNN'], bins=bin_dict[variable], alpha=0.8, edgecolor='black', linewidth=1.2,
                 label='$\mu$: %0.3f, $\sigma$: %0.3f Regression' % (mean_DNN, std_DNN))

        plt.legend()
        plt.title('Jet '+variable)
        plt.xlabel(variable)
        plt.ylabel('Jets')
        plt.yscale('log', nonposy='clip')
        plt.savefig('plots/'+variable+'_'+type+'_histogram.pdf')
        plt.clf()
