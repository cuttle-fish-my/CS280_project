## 1. Dataset Structure
Please download [[dataset](https://epan.shanghaitech.edu.cn/l/VFi1gQ)] and unzip files like the following structure:
```
COCO
    ├── annotations
    │   ├── train2017
    │   ├── val2017
    │   ├── stuff_train2017.json
    │   ├── stuff_val2017.json
    │   ├── thing_train2017.json
    │   └── thing_val2017.json
    ├── train2017
    ├── val2017
    ├── train_filenames.pkl
    ├── val_filenames.pkl
    └── categories.pkl
```
All the `*.pkl` files are generated by `datasets/COCO.py` utilizing `datasets/COCO/annotations/*.json`, but those `*.pkl`
files have already been generated for you and you do not need to regenerate it again.
## 2. Pretrained DDPM weight
Please download [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)]
and rename the downloaded `*.pt` file to `DDPM_pretrained.pt` and put it to path `CS280_project/checkpoints`.
## 3. Model Usage
- ### Single-GPU training
    ```bash
    python scripts/train.py --batch_size 2 --data_dir ./datasets/COCO --category_pickle ./datasets/COCO/categories.pkl --filename_pickle ./datasets/COCO/train_filenames.pkl --save_dir ./result --DDPM_dir ./checkpoints/DDPM_pretrained.pt
    ```
    where ``--batch_size`` should be determined by your GPU
- ### Multi-GPU training
    Before training, you need to ensure you have `torchrun` installed and `torchrun` is located in your virtual environment instead of system environment. 

    Check the location of `torchrun` with the following command:
    ```bash
    which torchrun
    ```
    if the output is like `/home/username/anaconda3/bin/torchrun`, then you need to reinstall `torch` in your virtual environment:
    ```bash
    pip install --ignore-installed torch
    ```
    After that, you can train the model with the following command:
    ```bash
    torchrun  --nproc-per-node $NUM_GPUS python scripts/train.py --batch_size $BATCH_SIZE --data_dir ./datasets/COCO --category_pickle ./datasets/COCO/categories.pkl --filename_pickle ./datasets/COCO/train_filenames.pkl --save_dir ./result --DDPM_dir ./checkpoints/DDPM_pretrained.pt
    ```
    where `$NUM_GPUS` is the number of GPUs you want to use and `$BATCH_SIZE` is the batch size for each GPU.

*Note:* If you encountered the error like:
``
ModuleNotFoundError: No module named 'improved_diffusion'
``
Then you have several options to solve this problem:
- ### Option1: 
    Add the path of `improved-diffusion` to `PYTHONPATH` in your terminal (need to be done every time you open a new terminal)
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```
- ### Option2:
    Add the following code to the beginning of `train.py` (already added) (permanent):
    ```python
    import os
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))
    ```
- ### Option3:
    Install `improved-diffusion` as a package in editable mode (permanent):
    ```bash
    pip install -e .
    ```

## 4. `mpi4py` installation on AI-cluster
If you failed when installing the package `mpi4py` on AI-cluster (`10.15.89.191/192`), you can try to install the package with the following options:
- ### Option 1: Install with pip
    The original installation with `pip` failed as it cannot find the `mpi` compilers. You can get the access of the compilers with following steps:
    ```bash
    mpi-selector-menu
    ```
    Then you can choose the `openmpi` compiler for `user`. After that, you can install the package with `pip`:
    ```bash
    pip install mpi4py
    ```
- ### Option 2: Install with conda (not tested)
    The installation with `conda` is much easier. You can install the package with the following command:
    ```bash
    conda install mpi4py
    ```
