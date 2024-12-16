# FusionMamba
- Code for the paper: "FusionMamba: Efficient Remote Sensing Image Fusion with State Space Model", 2024.

- First application of the state space model (SSM) in the hyper-spectral pansharpening and hyper-spectral image super-resolution (HISR) tasks.

- State-of-the-art (SOTA) performance in pansharpening, hyper-spectral pansharpening, and HISR tasks.

# Paper
- For a detailed understanding of our method, please refer to the paper: [FusionMamba: Efficient Remote Sensing Image Fusion with State Space Model](https://arxiv.org/abs/2404.07932).
- This paper has been published in the IEEE Transactions on Geoscience and Remote Sensing.

# Get Started
## Dataset
- Datasets for pansharpening: [PanCollection](https://github.com/liangjiandeng/PanCollection). We recommend downloading datasets in the h5py format. The testing toolbox can be found [here](https://github.com/liangjiandeng/DLPan-Toolbox).

- Datasets for hyper-spectral pansharpening: [HyperPanCollection](https://github.com/liangjiandeng/HyperPanCollection).
We recommend downloading datasets in the h5py format.

- Dataset for HISR: the CAVE dataset. You can find this dataset on the Internet.

## Installation
1. Clone the repository:
```
git clone https://github.com/PSRben/FusionMamba.git
```

2. Install the Mamba implementation by following the instructions in the Mamba-block directory.

3. Install other packages:
```
pip install einops h5py opencv-python torchinfo scipy numpy
```

## Usage
- This repository is only for the pansharpening task.

- The model weights trained on the WV3 dataset for 400 epochs can be found in the weights directory.

```
# train
python train.py --train_data_path ./path_to_data/train_WV3.h5 --val_data_path ./path_to_data/valid_WV3.h5
# test
python test.py --file_path ./path_to_data/name.h5 --save_dir ./path_to_dir --weight ./weights/epochs.pth
```

# Citation
```
@ARTICLE{10750233,
  author={Peng, Siran and Zhu, Xiangyu and Deng, Haoyu and Deng, Liang-Jian and Lei, Zhen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FusionMamba: Efficient Remote Sensing Image Fusion With State Space Model}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2024.3496073}}
```

# Contact
We are glad to hear from you. If you have any questions, please feel free to contact siran_peng@163.com.
