#!/bin/bash

# NCCL2
if [ ! -f "$HOME/Downloads/nccl_2.4.8-1+cuda10.0_x86_64.txz" ]; then
	echo Download nccl_2.4.8-1+cuda10.0_x86_64.txz from \"https://developer.nvidia.com/nccl/nccl-download\"
	echo Save it in \""$HOME/Downloads"\"
	exit 1
fi

tar -vxf "$HOME/Downloads/nccl_2.4.8-1+cuda10.0_x86_64.txz" -C "$HOME/Downloads"

sudo cp "$HOME/Downloads/nccl_2.4.8-1+cuda10.0_x86_64/lib/libnccl*" /usr/lib/x86_64-linux-gnu/
sudo cp "$HOME/Downloads/nccl_2.4.8-1+cuda10.0_x86_64/include/nccl.h"  /usr/include/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> "$HOME/.bashrc"
source "$HOME/.bashrc"

# Open MPI
if test -f /usr/bin/mpirun; then
    sudo mv /usr/bin/mpirun /usr/bin/bk_mpirun
fi

if test -f /usr/bin/mpirun.openmpi; then
    sudo mv /usr/bin/mpirun.openmpi /usr/bin/bk_mpirun.openmpi
fi


wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz -P "$HOME/Downloads"
tar -xvf "$HOME/Downloads/openmpi-4.0.1.tar.gz" -C "$HOME/Downloads"
cd "$HOME/Downloads/openmpi-4.0.1"
./configure --prefix=$HOME/openmpi
make -j 8 all
make install

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/openmpi/lib' >> "$HOME/.bashrc"
echo 'export PATH=$PATH:$HOME/openmpi/bin' >> "$HOME/.bashrc"
source "$HOME/.bashrc"

# Python env
cd $HOME

# # Install g++-4.8 (for running horovod with TensorFlow)
sudo apt install g++-4.8

# # Create a Python3.6 virtual environment
sudo apt-get install python3-pip
sudo pip3 install virtualenv

if [ -d "$HOME/venv-keras" ];
then
	rm -rf "$HOME/venv-keras"
fi

virtualenv -q -p /usr/bin/python3.6 "$HOME/venv-keras"
source "$HOME/venv-keras/bin/activate"

# Install keras and TensorFlow GPU backend
"$HOME/venv-keras/bin/pip" install tensorflow-gpu==1.13.2 keras

# Horovod
HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 "$HOME/venv-keras/bin/pip" install --no-cache-dir horovod