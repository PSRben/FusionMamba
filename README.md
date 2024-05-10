# FusionMamba: Efficient Image Fusion with State Space Model
- Code for the paper: "FusionMamba: Efficient Image Fusion with State Space Model", 2024.

- First application of the state space model (SSM) in the hyper-spectral pansharpening and hyper-spectral image super-resolution (HISR) tasks.

- State-of-the-art (SOTA) performance in pansharpening, hyper-spectral pansharpening, and HISR tasks.

# Paper
For a detailed understanding of our method, please refer to the paper: [FusionMamba: Efficient Image Fusion with State Space Model](https://arxiv.org/abs/2404.07932).


# Get Started
## Dataset
- Datasets for pansharpening: [PanCollection](https://github.com/liangjiandeng/PanCollection). 
We recommend downloading the dataset in h5py format.

- Datasets for hyper-spectral pansharpening: [HyperPanCollection](https://github.com/liangjiandeng/HyperPanCollection).
We recommend downloading the dataset in h5py format.

- Dataset for HISR: the CAVE dataset. You can find this dataset on the Internet.

## Installation
1. Clone the repository:
```
git clone https://github.com/PSRben/FusionMamba.git
```

2. Install the Mamba implementation by following the instructions in the Mamba-block folder.

3. Install other packages:
```
pip install einops h5py opencv-python torchinfo scipy numpy
```

## Usage
- This repository is only for the pansharpening task.

- The model weight trained on the WV3 dataset for 400 epochs can be found in the weights dir.

```
# train
python train.py --train_data_path ./path_to_data/train_WV3.h5 --val_data_path ./path_to_data/valid_WV3.h5
# test
python test.py --file_path ./path_to_data/name.h5 --save_dir ./path_to_dir --weight ./weights/epochs.pth
```

# Citation
```
@misc{peng2024fusionmamba,
      title={FusionMamba: Efficient Image Fusion with State Space Model}, 
      author={Siran Peng and Xiangyu Zhu and Haoyu Deng and Zhen Lei and Liang-Jian Deng},
      year={2024},
      eprint={2404.07932},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Contact
We are glad to hear from you. If you have any questions, please feel free to contact siran_peng@163.com.