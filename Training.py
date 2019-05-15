# Restrict to running on only one GPU on hefaistos
import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False
# ////////////////////////////////////////////////

# Needed to produce matplotlib plots on a machine without display
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend as K
K.set_session(sess)
import pandas as pd
from root_pandas import read_root, to_root
import numpy as np
import glob
import math
import sys
import re
import os
import shutil
import hickle as hkl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from joblib import dump

from Models import create_model
import Callbacks

import argparse

useGenParticles = False

parser = argparse.ArgumentParser()
parser.add_argument("--useGenParticles", help="Use gen particles instead of pfCandidates in the training", action="store_true")
args = parser.parse_args()
if args.useGenParticles:
	useGenParticles = True
	print "Using gen particles"


# (Re)create folder for plots
folders_ = ['plots','Graph']
for dir in folders_:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

# Lock the random seed for reproducibility
np.random.seed = 7

n_particles = 20
numbers = [str(x) for x in range(n_particles)]

path_to_files = "/work/hajohajo/JetRegression/preprocessed_genJets_and_pfJets_tmp/"
input_files = glob.glob(path_to_files+"*.root")

# A trick to easily read the input variable names and separate the neutral, charged and photons
dummy = read_root(input_files[0], 'tree', chunksize=1).__iter__().next()

Ccand_variables = list(dummy.filter(regex='jetPF_chg_'))
Ncand_variables = list(dummy.filter(regex='jetPF_neu_'))
Pcand_variables = list(dummy.filter(regex='jetPF_pho_'))
if useGenParticles:
	Ccand_variables = list(dummy.filter(regex='genPF_chg_'))
	Ncand_variables = list(dummy.filter(regex='genPF_neu_'))
	Pcand_variables = list(dummy.filter(regex='genPF_pho_'))

#Global_variables = list(set(list(dummy.filter(regex='jet')))-set(Ccand_variables+Ncand_variables+Pcand_variables))+list(dummy.filter(regex='QG_'))
Global_variables = ['jetPt','jetEta','QG_mult','QG_axis2','QG_ptD']
Gen_variables = list(dummy.filter(regex='genJet'))
Flavor_variables = list(dummy.filter(regex='isPhys'))

# Drops excessive pfCandidates, if there are more than n_particles present in the input files
Ccand_variables = [cand for cand in Ccand_variables if (any(num == cand.split("_")[-1] for num in numbers))]
Ncand_variables = [cand for cand in Ncand_variables if (any(num == cand.split("_")[-1] for num in numbers))]
Pcand_variables = [cand for cand in Pcand_variables if (any(num == cand.split("_")[-1] for num in numbers))]

# Reorganize pfCandidates so that they are in correct order to be fed into the LSTM layers after reshaping
for list_ in [Ccand_variables, Ncand_variables, Pcand_variables]:
    dummy_list = [re.split(r'(\d+)', s)[0:-1] for s in list_]
    dummy_list.sort(key=lambda l: (int(l[-1]), l[0]))
    list_ = ["".join(element) for element in dummy_list]

Training_variables = Ccand_variables + Ncand_variables + Pcand_variables + Global_variables

# Save variable names to guarantee same ordering when testing
hkl.dump([n_particles, Ccand_variables, Ncand_variables, Pcand_variables, Global_variables, Gen_variables, Flavor_variables, Training_variables],'Variables.hkl')

data = read_root(input_files, 'tree', columns=Training_variables+Gen_variables+Flavor_variables)
data = data.sample(frac=1.0)
data.reset_index(drop=True, inplace=True)

# Target for the regression to predict the correction factor
data['target'] = data.genJetPt/data.jetPt

# Additional selections to limit phase space
data = data[(np.abs(data.jetEta) < 1.3) & (data.genJetPt > 60.) & ((data.target > 0.9) & (data.target < 1.1))]

# Split into set used for training and validation, and a separate test sets 0.9/0.1
training, test = train_test_split(data, shuffle=True, test_size=0.1)
test.reset_index(drop=True, inplace=True)
training.reset_index(drop=True, inplace=True)

# Save test data to a separate file for post training plotting
to_root(test, 'test_data.root', key='tree')

# Scale input variables for training and save scaler for future use in plotting
scaler = MinMaxScaler().fit(training[Training_variables].values)
dump(scaler, "scaler.pkl")
train_inp = pd.DataFrame(scaler.transform(training[Training_variables].values), columns=Training_variables)
train_trg = training['target']

# Prepare test data for monitoring plots
test_true = test[['isPhysUDS', 'isPhysG', 'genJetPt', 'jetPt']]
test_inp = pd.DataFrame(scaler.transform(test[Training_variables].values), columns=Training_variables)
test_Ccands = test_inp[Ccand_variables]
test_Ncands = test_inp[Ncand_variables]
test_Pcands = test_inp[Pcand_variables]
test_Globals = test_inp[Global_variables]

test_Ccands = np.reshape(np.array(test_Ccands),(test_Ccands.shape[0], n_particles, test_Ccands.shape[1]/n_particles))
test_Ncands = np.reshape(np.array(test_Ncands),(test_Ncands.shape[0], n_particles, test_Ncands.shape[1]/n_particles))
test_Pcands = np.reshape(np.array(test_Pcands),(test_Pcands.shape[0], n_particles, test_Pcands.shape[1]/n_particles))


# Separate globals, charged, neutral and photon candidates to their own inputs for the training
train_Ccands = train_inp[Ccand_variables]
train_Ncands = train_inp[Ncand_variables]
train_Pcands = train_inp[Pcand_variables]
train_Globals = train_inp[Global_variables]

# Reshaping pfCand arrays to fit LSTMs
train_Ccands = np.reshape(np.array(train_Ccands),(train_Ccands.shape[0], n_particles, train_Ccands.shape[1]/n_particles))
train_Ncands = np.reshape(np.array(train_Ncands),(train_Ncands.shape[0], n_particles, train_Ncands.shape[1]/n_particles))
train_Pcands = np.reshape(np.array(train_Pcands),(train_Pcands.shape[0], n_particles, train_Pcands.shape[1]/n_particles))

# Create model
model = create_model('ResNet', train_Ccands.shape, train_Ncands.shape, train_Pcands.shape, train_Globals.shape)
callbacks = Callbacks.getStandardCallbacks()
callbacks += Callbacks.makePlots([test_Ccands, test_Ncands, test_Pcands, test_Globals], test_true, 'plots')
print model.summary()


print test_Globals.shape
#Perform training
model.fit([train_Ccands, train_Ncands, train_Pcands, train_Globals],
          train_trg,
          batch_size=256,
          validation_split=0.1,
          shuffle=True,
          epochs=500,
          callbacks=callbacks)

model.save('Trained_model.h5')
