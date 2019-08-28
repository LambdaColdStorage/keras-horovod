# Keras Demo

### Installation

```
git clone https://github.com/chuanli11/navy.git
cd navy

# Install python env 
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
virtualenv -p /usr/bin/python3.6 venv-keras
. venv-keras/bin/activate

pip install tensorflow-gpu==1.13.2 keras

# Install Openmpi
mkdir openmpi
cd openmpi
wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.gz
tar -xvf openmpi-3.1.2.tar.gz
cd openmpi-3.1.2
./configure --prefix=$HOME/openmpi
make -j 8 all
make install

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/openmpi/lib' >> ~/.bashrc
echo 'export PATH=$PATH:~/openmpi/bin' >> ~/.bashrc
cd ../..

# Need to make sure there is no /usr/bin/miprun

# Install NCCL2

# Download from NVIDIA's website: https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.4/prod//nccl_2.4.8-1%2Bcuda10.0_x86_64.txz
tar -vxf nccl_2.4.8-1+cuda10.0_x86_64.txz

sudo cp nccl_2.4.8-1+cuda10.0_x86_64/lib/libnccl* /usr/lib/x86_64-linux-gnu/
sudo cp nccl_2.4.8-1+cuda10.0_x86_64/include/nccl.h  /usr/include/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc

# Install g++-4.8 (for running horovod with TensorFlow)
sudo apt install g++-4.8


# Install Horovod
HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod
```


### Multi-GPU training

```
horovodrun -np 2 -H localhost:2 python keras_mnist.py

mpirun -np 2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python keras_mnist.py
```