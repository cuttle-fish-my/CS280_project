## `mpi4py` installation on AI-cluster
If you failed when installing the package `mpi4py` on AI-cluster (`10.15.89.191/192`), you can try to install the package with the following options:
### Option 1: Install with pip
The original installation with `pip` failed as it cannot find the `mpi` compilers. You can get the access of the compilers with following steps:
```bash
mpi-selector-menu
```
Then you can choose the `openmpi` compiler for `user`. After that, you can install the package with `pip`:
```bash
pip install mpi4py
```
### Option 2: Install with conda (not tested)
The installation with `conda` is much easier. You can install the package with the following command:
```bash
conda install mpi4py
```
