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
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
tar -xvf openmpi-4.0.0.tar.gz
cd openmpi-4.0.0
./configure --prefix=$HOME/openmpi
make -j 8 all
make install

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/openmpi/lib' >> ~/.bashrc
echo 'export PATH=$PATH:~/openmpi/bin' >> ~/.bashrc
cd ../..


or 

wget https://s3-us-west-2.amazonaws.com/lambdalabs-files/openmpi_4.0.0-2_amd64.deb
sudo dpkg -i openmpi_4.0.0-2_amd64.deb
sudo apt install libopenmpi-dev


# Install NCCL2

# Download from NVIDIA's website: https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.4/prod//nccl_2.4.8-1%2Bcuda10.0_x86_64.txz
tar -vxf nccl_2.4.8-1+cuda10.0_x86_64.txz

sudo cp nccl_2.4.8-1+cuda10.0_x86_64/lib/libnccl* /usr/lib/x86_64-linux-gnu/
sudo cp nccl_2.4.8-1+cuda10.0_x86_64/include/nccl.h  /usr/include/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc

# Install g++-4.8 (for running horovod with TensorFlow)
sudo apt install g++-4.8


# Install Horovod
HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
```


### Multi-GPU training

```
mpirun -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python keras_mnist.py
```