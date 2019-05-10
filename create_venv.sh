#!/bin/bash

# Create the temporary enviornment on KTH's lab computers

TMP_ENV=/tmp/chrislin_pytorch

rm -rf ${TMP_ENV}

# create the venv environment
virtualenv -p /usr/bin/python3 ${TMP_ENV}

# activate the environment
source ${TMP_ENV}/bin/activate

# Update pip
pip install --upgrade pip

cd /tmp
rm -f torch-1.1.0-cp35-cp35m-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
pip install torch-1.1.0-cp35-cp35m-linux_x86_64.whl
pip install torchvision

echo "Please activate via the command: source ${TMP_ENV}/bin/activate"