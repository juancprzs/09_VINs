# 09_VINs
Value Iteration Networks demo and explanation

This is, pretty much, a copy and paste from [Kent Sommer's repo](https://github.com/kentsommer/pytorch-value-iteration-networks/blob/master/README.md) on this .

(Thank you, Kent Sommer!)

# VIN: [Value Iteration Networks](https://arxiv.org/abs/1602.02867)

## Installation
This repository requires following packages:
- [SciPy](https://www.scipy.org/install.html) >= 0.19.0
- [Python](https://www.python.org/) >= 2.7 (if using Python 3.x: python3-tk should be installed)
- [Numpy](https://pypi.python.org/pypi/numpy) >= 1.12.1
- [Matplotlib](https://matplotlib.org/users/installing.html) >= 2.0.0
- [PyTorch](http://pytorch.org/) >= 0.1.11

Use `pip` to install the necessary dependencies:
```
pip install -U -r requirements.txt 
```
Note that PyTorch cannot be installed directly from PyPI; refer to http://pytorch.org/ for custom installation instructions specific to your needs. 
## How to train
#### 8x8 gridworld
```bash
python train.py --datafile dataset/gridworld_8x8.npz --imsize 8 --lr 0.005 --epochs 30 --k 10 --batch_size 128
```
#### 16x16 gridworld
```bash
python train.py --datafile dataset/gridworld_16x16.npz --imsize 16 --lr 0.002 --epochs 30 --k 20 --batch_size 128
```
#### 28x28 gridworld
```bash
python train.py --datafile dataset/gridworld_28x28.npz --imsize 28 --lr 0.002 --epochs 30 --k 36 --batch_size 128
```
**Flags**: 
- `datafile`: The path to the data files.
- `imsize`: The size of input images. One of: [8, 16, 28]
- `lr`: Learning rate with RMSProp optimizer. Recommended: [0.01, 0.005, 0.002, 0.001]
- `epochs`: Number of epochs to train. Default: 30
- `k`: Number of Value Iterations. Recommended: [10 for 8x8, 20 for 16x16, 36 for 28x28]
- `l_i`: Number of channels in input layer. Default: 2, i.e. obstacles image and goal image.
- `l_h`: Number of channels in first convolutional layer. Default: 150, described in paper.
- `l_q`: Number of channels in q layer (~actions) in VI-module. Default: 10, described in paper.
- `batch_size`: Batch size. Default: 128

## How to test / visualize paths (requires training first)
#### 8x8 gridworld
```bash
python test.py --weights trained/vin_8x8.pth --imsize 8 --k 10
```
#### 16x16 gridworld
```bash
python test.py --weights trained/vin_16x16.pth --imsize 16 --k 20
```
#### 28x28 gridworld
```bash
python test.py --weights trained/vin_28x28.pth --imsize 28 --k 36
```
To visualize the optimal and predicted paths simply pass:
```bash 
--plot
```

**Flags**: 
- `weights`: Path to trained weights.
- `imsize`: The size of input images. One of: [8, 16, 28]
- `plot`: If supplied, the optimal and predicted paths will be plotted 
- `k`: Number of Value Iterations. Recommended: [10 for 8x8, 20 for 16x16, 36 for 28x28]
- `l_i`: Number of channels in input layer. Default: 2, i.e. obstacles image and goal image.
- `l_h`: Number of channels in first convolutional layer. Default: 150, described in paper.
- `l_q`: Number of channels in q layer (~actions) in VI-module. Default: 10, described in paper.

## Results
Gridworld | Sample One | Sample Two
-- | --- | ---
8x8 | <img src="results/8x8_1.png" width="450"> | <img src="results/8x8_2.png" width="450">
16x16 | <img src="results/16x16_1.png" width="450"> | <img src="results/16x16_2.png" width="450">
28x28 | <img src="results/28x28_1.png" width="450"> | <img src="results/28x28_2.png" width="450">

## Datasets
Each data sample consists of an obstacle image and a goal image followed by the (x, y) coordinates of current state in the gridworld. 

Dataset size | 8x8 | 16x16 | 28x28
-- | -- | -- | --
Train set | 81337 | 456309 | 1529584
Test set | 13846 | 77203 | 251755

The datasets (8x8, 16x16, and 28x28) included in this repository can be reproduced using the ```dataset/make_training_data.py``` script. Note that this script is not optimized and runs rather slowly (also uses a lot of memory :D)

## HomeWork
Train and evaluate the code in three new domains of the grid-world (three new -square- dimensions for the board) and plot the success rates as a function of the domain size.
