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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from joblib import dump

from Models import create_model
import Callbacks

# Lock the random seed for reproducibility
np.random.seed = 7

n_particles = 50

#input_files = glob.glob("/work/hajohajo/JetRegStudy/preprocessed_genJets_and_pfJets/*.root")
input_files = glob.glob("/work/hajohajo/REMOVE/JetRegression/preprocessed_genJets_and_pfJets/*.root")

# A trick to easily read the input variable names and separate the neutral, charged and photons
dummy = read_root(input_files[0], 'tree', chunksize=1).__iter__().next()

Ccand_variables = list(dummy.filter(regex='jetPF_chg_'))
Ncand_variables = list(dummy.filter(regex='jetPF_neu_'))
Pcand_variables = list(dummy.filter(regex='jetPF_pho_'))
Global_variables = list(set(list(dummy.filter(regex='jet')))-set(Ccand_variables+Ncand_variables+Pcand_variables))
Gen_variables = list(dummy.filter(regex='genJet'))
Flavor_variables = list(dummy.filter(regex='isPhys'))

# Reorganize pfCandidates so that they are in correct order to be fed into the LSTM layers after reshaping
for list_ in [Ccand_variables, Ncand_variables, Pcand_variables]:
    dummy_list = [re.split(r'(\d+)', s)[0:-1] for s in list_]
    dummy_list.sort(key=lambda l: (int(l[-1]), l[0]))
    list_ = ["".join(element) for element in dummy_list]

Training_variables = Ccand_variables + Ncand_variables + Pcand_variables + Global_variables
data = read_root(input_files, 'tree', columns=Training_variables+Gen_variables+Flavor_variables)

# Target for the regression to predict the correction factor
data['target'] = data.genJetPt/data.jetPt

# Additional selections to limit phase space
data = data[(np.abs(data.jetEta) < 1.3) & (data.genJetPt > 60.) ] #& ((data.target > 0.85) & (data.target < 1.25))]

# Split into set used for training and validation, and a separate test sets 0.9/0.1
training, test = train_test_split(data, shuffle=True, test_size=0.1)

# Save test data to a separate file for post training plotting
to_root(test, 'test_data.root', key='tree')

# Scale input variables for training and save scaler for future use in plotting
scaler = StandardScaler().fit(training[Training_variables])
dump(scaler, "scaler.pkl")
train_inp = pd.DataFrame(scaler.transform(training[Training_variables]), columns=Training_variables)
train_trg = training['target']

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
model = create_model('DeepJet', train_Ccands.shape, train_Ncands.shape, train_Pcands.shape, train_Globals.shape)

callbacks = Callbacks.get_callbacks()

#Perform training
model.fit([train_Ccands, train_Ncands, train_Pcands, train_Globals],
          train_trg,
          batch_size=1024,
          validation_split=0.1,
          shuffle=True,
          epochs=500,
          callbacks=callbacks)

model.save('Trained_model.h5')
