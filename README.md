# 8 x MI300X Testing 

This repo tracks the testing of an 8 x MI300X node graciously provided by [Hot Aisle](https://hotaisle.xyz/) for [Benchmark & Analysis](https://hotaisle.xyz/benchmarks-and-analysis/) testing. If you like the results, you can of course, [rent the exact same setup](https://hotaisle.xyz/pricing/) from them.

To better organize the testing and results, I've moved all testing into [Jupyter](https://jupyter.org/) notebooks and published with [Quarto](https://quarto.org/). You can of course load the raw ipynbs, or the published HTML.

## Install
If you are looking to run the Jupyter/Quarto notebooks:
```
# First install mamba
# https://github.com/conda-forge/miniforge?tab=readme-ov-file#install
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p PREFIX /<INSTALL_PATH>/miniforge3
mamba init

# In (base) env
mamba activate base
pip install jupyter jupyterlab
mamba install nb_conda_kernels

# make sure you include `ipykernel` in each venv that you want to use with jupyter

# https://quarto.org/docs/get-started/
# install Quarto for your platform, eg
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.57/quarto-1.5.57-linux-amd64.deb
sudo dpkg -i quarto-1.5.57-linux-amd64.deb

# For Quarto integration
pip install jupyterlab-quarto

# Run Jupyter if you want to run/edit the files
jupyter lab
```

- [00-system-info.ipynb](00-system-info.ipynb) - for info about the node



# 
llama.cpp
---
time make GGML_HIPBLAS=1 -j208

(base) hotaisle@ENC1-CLS01-SVR09:~/llama.cpp$ ./llama-bench -m /mnt/nvme0n1p1/llama-2-7b.Q4_0.gguf 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 8 ROCm devices:
  Device 0: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 1: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 2: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 3: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 4: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 5: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 6: AMD Instinct MI300X, compute capability 9.4, VMM: no
  Device 7: AMD Instinct MI300X, compute capability 9.4, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         pp512 |       1333.08 ± 4.99 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         tg128 |        174.99 ± 2.20 |

build: d5cb8684 (3891)

HIP_VISIBLE_DEVICES=0 time ./llama-bench -m /mnt/nvme0n1p1/llama-2-7b.Q4_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI300X, compute capability 9.4, VMM: no
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         pp512 |      1334.37 ± 12.73 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |         tg128 |        183.18 ± 0.77 |

build: d5cb8684 (3891)
8.51user 1.52system 0:08.57elapsed 117%CPU (0avgtext+0avgdata 5281008maxresident)k
0inputs+12256outputs (1major+496848minor)pagefaults 0swaps

$ HIP_VISIBLE_DEVICES=0 time ./llama-bench -m /mnt/nvme0n1p1/llama-2-7b.Q4_0.gguf -fa 1
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI300X, compute capability 9.4, VMM: no
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |         pp512 |       1272.03 ± 7.97 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |         tg128 |        157.84 ± 0.35 |

build: d5cb8684 (3891)
9.13user 1.43system 0:09.23elapsed 114%CPU (0avgtext+0avgdata 5283472maxresident)k
0inputs+12256outputs (1major+496366minor)pagefaults 0swaps


https://github.com/ggerganov/llama.cpp/pull/8082
https://github.com/ggerganov/llama.cpp/pull/8082/files



mamba create -n llm python=3.11
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
pip install transformers
pip instal flash-attn
pip install torchao
pip install torchtune
pip install torchchat

git clone https://github.com/pytorch/torchchat.git
cd torchchat
python3 -m venv .venv
source .venv/bin/activate
./install/install_requirements.sh


git clone https://github.com/ROCm/composable_kernel.git && \
cd composable_kernel && \
mkdir build && \
cd build

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D GPU_TARGETS="gfx942"                                                                    \
..
