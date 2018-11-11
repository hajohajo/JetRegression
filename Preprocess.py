import pandas as pd
import root_pandas
import os
import shutil
import glob
from keras_preprocessing.sequence import pad_sequences

#Folder containing input root files produced for example with Kimmo's Jetter
path = '~/QCD_tuples_withPF_v5/'
fnames = glob.glob(path + '*.root')

#Global jet variables to be stored
globals = ['genJetPt', 'genJetEta', 'genJetPhi', 'genJetMass',
            'physFlav', 'jetPt', 'jetRawPt', 'jetEta', 'jetPhi',
            'QG_ptD', 'QG_axis2', 'QG_mult', 'isPhysUDS','isPhysG']

#Particle level variables to be stored from each jet
particles = ['genPF_neu_pT', 'genPF_neu_dPhi', 'genPF_neu_dTheta', 'genPF_neu_dR',
        'genPF_pho_pT', 'genPF_pho_dPhi', 'genPF_pho_dTheta', 'genPF_pho_dR',
        'genPF_chg_pT', 'genPF_chg_dPhi', 'genPF_chg_dTheta', 'genPF_chg_dR',
        'jetPF_chg_pT', 'jetPF_chg_dPhi', 'jetPF_chg_dTheta', 'jetPF_chg_dR',
        'jetPF_chg_puppiW', 'jetPF_chg_vtxAssQ', 'jetPF_chg_pTrel', 'jetPF_chg_dz',
        'jetPF_chg_dxy', 'jetPF_neu_pT', 'jetPF_neu_dPhi', 'jetPF_neu_dR', 'jetPF_neu_dTheta',
        'jetPF_pho_pT', 'jetPF_pho_dPhi', 'jetPF_pho_dR', 'jetPF_pho_dTheta']

#Number of particles to be considered
n_particles = 50

out_folder = "preprocessed_genJets_and_pfJets"

#Recreate the output folder
if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
os.makedirs(out_folder)

counter = 1
for file in fnames:
    print("Processing %d file", counter)
    for df in root_pandas.read_root(file, 'jetTree', columns = globals + particles, chunksize=50000):
        #Initial selections to focus on interesting subjets, makes preprocessing faster
        df = df[(df.isPhysUDS==1)|(df.isPhysG==1)]
        df = df[(df.genJetEta < 2.4)]
        df.reset_index(drop=True, inplace=True)

        df2=pd.DataFrame()
        column_names = [column+'_'+str(x) for x in range(n_particles)]

        #Crazy compact flattening/unrolling the particle level variables stored as vectors in the jets
        for column in particles:
            df2=pd.concat([df2, pd.DataFrame(pd.DataFrame(pad_sequences(df[column].tolist(),
                                                    maxlen=n_particles, padding='post', dtype='float64'),
                                                    columns=column_names))], axis=1)
        df=pd.concat([df[globals], df2])

        #Store output files to out_folder in files containing chunksize jets each
        save_name=out_folder + '/preprocessed_' + str(counter) + '.root'
        df.to_root(save_name, key = 'tree')
        counter=counter+1