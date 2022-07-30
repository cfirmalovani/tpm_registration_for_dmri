#!/usr/bin/env bash

echo "Creating Conda Environment"
conda create --name tpm_reg python=3.8
conda activate tpm_reg

echo "Installing PIP libraries"
pip install dipy==1.1.1
pip install nibabel==3.1.0
pip install numpy
pip install scipy==1.4.1
pip install xgboost==0.90
pip install Cython==0.29.17

echo "Installing scikit-learn"
git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
git checkout 0.21.X
export SKLEARN_NO_OPENMP=TRUE
python setup.py install

echo "Clean up"
cd ..
rm -r scikit-learn
rmdor scikit-learn

echo "Compiling Cython files"
rm multimodal_crosscorr/*.so
rm src/mod_dipy/*.so
cd multimodal_crosscorr
python setup.py build_ext --inplace
SO_FILE=$(ls *.so)
cp $SO_FILE ../src/mod_dipy/
