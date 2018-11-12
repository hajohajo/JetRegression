#SETTING UP ENVIRONMENT

###-----------------------------------------
###If you don't have root installed, fetch the binary and set it up to .bashrc
cd ~
wget https://root.cern.ch/download/root_v6.14.04.Linux-ubuntu18-x86_64-gcc7.3.tar.gz
tar -xzf root_v6.14.04.Linux-ubuntu18-x86_64-gcc7.3.tar.gz
rm root_v6.14.04.Linux-ubuntu18-x86_64-gcc7.3.tar.gz
echo -e "source ~/root/bin/thisroot.sh \n\n$(cat .bashrc)" > .bashrc
source ~/root/bin/thisroot.sh
###-----------------------------------------


###Create virtual environment for your work and activate it
cd /*To your work directory*/
mkdir JetRegression
cd JetRegression
python3 -m venv virtual_environment

#Add paths to ROOT libraries so root_numpy and root_pandas will work. Same for CUDA. These can be added to the virtual_environment/bin/activate file
export PYTHONPATH=/work/hajohajo/vTest/bin/:$PYTHONPATH
export LD_LIBRARY_PATH=/work/hajohajo/vTest/bin/:$LD_LIBRARY_PATH
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
