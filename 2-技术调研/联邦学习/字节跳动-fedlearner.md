# Fedlearner™

https://github.com/bytedance/fedlearner

Fedlearner is collaborative machine learning framework that enables **joint modeling of data distributed between institutions**.

---

### Quick Start with Fedlearner

#### Install on local machine for testing

First clone the latest code of this library from github:

```
git clone https://github.com/bytedance/fedlearner.git --recursive
```

Then setup python environment to run the package. Make sure that you have Python 3.6, other versions may have dependency issues:

```
cd fedlearner
pip install -r requirements.txt
export PYTHONPATH=$(PWD):$PYTHONPATH
make protobuf
```

If you see errors during gmpy2 installation, you may need to install the GMP library first. Try

```
apt-get install libgmp-dev libmpc-dev libmpfr-dev
```

### Run Example

There are two ways to run a simple training example locally:

- run test.sh

```
cd example/mnist

./test.sh
```

- run it manually and view summary from TensorBoard

```
cd example/mnist

python make_data.py
python leader.py --local-addr=localhost:50051 --peer-addr=localhost:50052 --data-path=data/leader --checkpoint-path=log/checkpoint --save-checkpoint-steps=10 --summary-path=log/summary --summary-save-steps=10 &
python follower.py --local-addr=localhost:50052 --peer-addr=localhost:50051 --data-path=data/follower/ --checkpoint-path=log/checkpoint --save-checkpoint-steps=10 --summary-path=log/summary --summary-save-steps=10
tensorboard --logdir=log
```

For better display, run the last two commands in two different terminals.

---

### Deploying Fedlearner on a Kubernetes Cluster

Fedlearner is not just a model trainer. It also comes with surrounding infrastructures for cluster management, job management, job monitoring, and network proxies. This tutorial walks through the steps to deploy Fedlearner's job scheduler on a Kubernetes cluster and submit model training jobs.

You will have two options: use a real K8s cluster or use local mock K8s cluster with minikube. The later is recommended for local testing.

#### Setup a K8s Cluster

To setup a local K8s cluster for testing, you fist need to install minikube. See [this](https://kubernetes.io/docs/tasks/tools/install-minikube/) page for instructions on minikube installation. Minikube requires a VM driver. We recommend hyperkit or docker as vm-driver for minikube.

After installation, run the following command to start minikube with 32 cores and 8GB of memory.

```
# replace DRIVER with hyperkit, docker, or other VMs you installed.
minikube start --cpus=32 --memory=8Gi --vm-driver=DRIVER
```