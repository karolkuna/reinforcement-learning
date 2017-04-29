# Introduction
This is an open source project built for experimenting with deep reinforcement learning algorithms in environments with continuous action domain (e.g. robot control tasks) from OpenAI Gym. It is a part of my Master's thesis focusing on model-based deep reinforcement learning. It also includes TensorFlow implementation of [Deep Deterministic Policy Gradient algorithm](https://arxiv.org/pdf/1509.02971.pdf) and [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952v4.pdf).

# Deep Model-Learning Actor-Critic
DeepMLAC is a novel model-based actor-critic off-policy deep reinforcement learning algorithm inspired by [Dyna-MLAC](http://ieeexplore.ieee.org/document/7423912/). It is designed to work in deterministic environments with continuous action domains. DeepMLAC learns a model of the environment from the experience of interacting with the environment. Policy is learned using the model in an actor-model-critic setting. Model is also used for n-step temporal difference learning of value function. Policy, model, and value functions are approximated with fully connected neural networks and trained with minibatches selected from prioritized experience replay.

# Installation
(Tested with Python 2.7.12 + Ubuntu 16.04 + TensorFlow 1.1 + CUDA 8.0)

WARNING: In order to render OpenAI Gym environments inside Jupyter notebooks, you have to install NVIDIA drivers with --no-opengl-files option, i.e. <code>./NVIDIA-Linux-x86-375.39.run --no-opengl-files</code>. If you already have NVIDIA drivers with opengl libs installed, you have to uninstall them first.

1. Install TensorFlow with GPU support https://www.tensorflow.org/install/install_linux
2. Install dependencies <code>apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig</code>
3. Install OpenAI Gym <code>pip install gym[all]</code> https://gym.openai.com/docs
4. Install Jupyter <code>pip install jupyter</code>
5. Launch Jupyter notebook server with a virtual screen buffer <code>xvfb-run -s "-screen 0 1400x900x24" jupyter notebook</code>
6. Open Notebook Dashboard in web browser (https://localhost:8888) and run .ipynb file of your choosing http://jupyter.readthedocs.io/en/latest/running.html#running


