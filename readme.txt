#SETTING UP ENVIRONMENT

###-----------------------------------------
###If you don't have root with pyROOT installed and the environment setup, you can use the globally available one
echo -e "source /work/data/root/bin/thisroot.sh \n\n$(cat .bashrc)" > .bashrc
source /work/data/root/bin/thisroot.sh
###-----------------------------------------


###Create virtual environment for your work and activate it
cd /*To your work directory*/
git clone git@github.com:hajohajo/JetRegression.git
#mkdir JetRegression
cd JetRegression
python2 -m virtualenv virtual_environment

#Add paths to ROOT libraries so root_numpy and root_pandas will work. Same for CUDA. These can be added to the virtual_environment/bin/activate file
export PYTHONPATH=/work/data/root/bin/:$PYTHONPATH
export LD_LIBRARY_PATH=/work/data/root/bin/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

source virtual_environment/bin/activate

###Install necessary packages to your virtual environment
pip install -r requirements.txt


#PREPROCESSING JET TUPLES
Goal of this part is mostly to convert CMSSW outputs into a flattened jet per row format that can be easily handled
in Machine Learning. Example tool for generating this original CMSSW outputs can be found in Kimmo's jetter for example.

In Preprocess.py, set the path, globals, particles, n_particles and out_folder variables to suit your needs.

#TRAINING REGRESSION NETWORK


#PRODUCING PLOTS

# MONITORING TRAINING
If the callback tensorboard is used, a folder Graph will be created and monitoring can be done using

tensorboard --logdir path_to_current_dir/Graph 
