# Guide for Horovod Installation on a Lambda Machine

# Quick Start

__Step One: Download NCCL2__ 

Download NCCL to `~/Downloads`. Use v2.4.8, for CUDA 10.0, July 31,2019(O/S agnostic local installer, free to register).
https://developer.nvidia.com/nccl/nccl-download


__Step Two: Install Horovod and Everything__ 

```
./install.sh
```

__Step Three: Run multi-GPU training__

```
cd
. venv-keras/bin/activate
git clone https://github.com/horovod/horovod.git
cd horovod/examples
horovodrun -np 2 -H localhost:2 --mpi python keras_mnist.py
```


# Explaination


#### NCCL2

Download NCCL v2.4.8, for CUDA 10.0, July 31,2019(O/S agnostic local installer, free to register): 
https://developer.nvidia.com/nccl/nccl-download


```
tar -vxf ~/Downloads/nccl_2.4.8-1+cuda10.0_x86_64.txz -C ~/Downloads/

sudo cp ~/Downloads/nccl_2.4.8-1+cuda10.0_x86_64/lib/libnccl* /usr/lib/x86_64-linux-gnu/
sudo cp ~/Downloads/nccl_2.4.8-1+cuda10.0_x86_64/include/nccl.h  /usr/include/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc
source ~/.bashrc


```

#### Open MPI (Optional)

You do not need to do this step. Then GLOO will be used for message passing, in which case you can not use the 'mpirun' API to run training job.

To install Open MPI, you need to first check if there is already an old version of 'mpirun' on the machine. Remove it if there is one:
```
sudo mv /usr/bin/mpirun /usr/bin/bk_mpirun
sudo mv /usr/bin/mpirun.openmpi /usr/bin/bk_mpirun.openmpi
```

Use the following steps to install Open MPI:

```
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz -P ~/Downloads
tar -xvf ~/Downloads/openmpi-4.0.1.tar.gz -C ~/Downloads
cd ~/Downloads/openmpi-4.0.1
./configure --prefix=$HOME/openmpi
make -j 8 all
make install

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/openmpi/lib' >> ~/.bashrc
echo 'export PATH=$PATH:~/openmpi/bin' >> ~/.bashrc
source ~/.bashrc
```

#### Horovod

```
cd

# Install g++-4.8 (for running horovod with TensorFlow)
sudo apt install g++-4.8

# Create a Python3.6 virtual environment
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
virtualenv -p /usr/bin/python3.6 venv-keras
. venv-keras/bin/activate

# Install keras and TensorFlow GPU backend
pip install tensorflow-gpu==1.13.2 keras

HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod
```

# Keras Multi-GPU Training Using Horovod


#### Clone The Example Repo
```
git clone https://github.com/horovod/horovod.git
cd horovod/examples
```

#### With GLOO
```
horovodrun -np 2 -H localhost:2 python keras_mnist.py
```


#### With Open MPI

```
horovodrun -np 2 -H localhost:2 --mpi python keras_mnist.py
```

or

```
mpirun -np 2 \
    -H localhost:2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python keras_mnist.py
```