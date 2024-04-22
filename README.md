# ✨ 🐴 🔥 Official PONITA implementation for Pytorch with minimal library dependencies

See [the original github repo](https://github.com/ebekkers/ponita) for a [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) implementation. The original repo has more features than this one. The current repository is a minimal dependency implementation that currently only implements the fiber bundle method. Moreover, the dataloaders do not rely on PyTorch Geometric, but provide the same type of objects (graphs consisting of the tensors: x, pos, edge_index, batch).

## Conda environment
In order to run the code in this repository install the following conda environment
```
conda create --yes --name ponita-torch python=3.12 numpy
conda activate ponita-torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install tqdm
pip install rdkit
pip install pandas
pip3 install pytorch_lightning
pip3 install wandb
```
