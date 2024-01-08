#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh && conda init

conda create -n pytorch python=3.9
conda activate pytorch
pip install -r env/requirements_pytorch.txt 
pip install -e .
conda deactivate

conda create python=3.8 --name baseline_tf1 -y
conda activate baseline_tf1
pip install nvidia-pyindex
pip install -r env/requirements_tf1.txt
mkdir -p third-party && cd third-party
git clone https://github.com/onnx/onnx-tensorflow.git
cd onnx-tensorflow
git checkout 0e4f4836 # v1.7.0-tf-1.15m
git apply ../../env/onnx_tf.patch
pip install -e .
conda deactivate

conda create -n tensorflow python=3.7
conda activate tensorflow
pip install -r env/requirements_tf2.txt 
pip install -e .
conda deactivate

conda create python=3.8 --name baseline_jax -y
conda activate jax
pip install -r env/requirements_jax.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -f https://download.pytorch.org/whl/torch_stable.html
conda deactivate


