This is an open source project built for experimenting with reinforcement learning algorithms in environments with continuous action domain (e.g. robot control tasks) from OpenAI gym. It is a part of my Master's thesis focusing on model-based deep reinforcement learning.

# Deep Model-Learning Actor-Critic
DMLAC is a novel model-based off-policy deep reinforcement learning algorithm. It is designed to work in deterministic environments with continuous space and action domains.

# Installation
(Tested with Python 2.7.12 + Ubuntu 16.04 + TensorFlow 1.0 + CUDA 8.0)

WARNING: In order to render OpenAI Gym environments inside Jupyter notebooks, you have to install NVIDIA drivers with --no-opengl-files option, i.e. <code>./NVIDIA-Linux-x86-375.39.run --no-opengl-files</code>. If you already have NVIDIA drivers with opengl libs installed, you have to uninstall them first.

1. Install TensorFlow with GPU support https://www.tensorflow.org/install/install_linux
2. Install dependencies <code>apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig</code>
3. Install OpenAI Gym <code>pip install gym[all]</code> https://gym.openai.com/docs
4. Install Jupyter <code>pip install jupyter</code>
5. Launch Jupyter notebook server with a virtual screen buffer <code>xvfb-run -s "-screen 0 1400x900x24" jupyter notebook</code>
6. Open Notebook Dashboard in web browser (https://localhost:8888) and run .ipynb file of your choosing http://jupyter.readthedocs.io/en/latest/running.html#running


