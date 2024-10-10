# HNA-NAS: Hardware-aware Nested Architecture Search for Infrared Small Target Segmentation

## Algorithm Introduction

HNA-NAS: Hardware-aware Nested Architecture Search for Infrared Small Target Segmentation, Boyang Li, Miao Li, Yingqian Wang, Ting Liu, Zaiping Lin, Wei An, Weidong Sheng, Yulan Guo.

We propose a hardware-aware nested architecture NAS method (namely, HNA-NAS) to search for efficient IRST segmentation network. Different from previous NAS methods, our method adopt IRST specialized nested architecture as backbone and design a two-stage neural architecture search algorithm to progressively achieve architecture simplification and cell optimization. Both efficient nested architecture and hardware-friendly convolutional operation are searched to achieve the trade-off between segmentation performance and inference latency. Experimental results on three benchmark datasets show that the searched HNANAS can achieve comparable segmentation performance with much smaller number of parameters, much lower FLOPs, and much shorter inference latency on the CPU, GPU, Edge CPU, and Edge GPU devices


## Prerequisite
* Tested on Ubuntu 16.04, with Python 3.7, PyTorch 1.7, Torchvision 0.8.1, CUDA 11.1, and 1x NVIDIA 3090 and also 

* [The NUDT-SIRST download dir](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt) (Extraction Code: nudt)

* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst) [[ACM]](https://arxiv.org/pdf/2009.14530.pdf)

## Usage

#### On Ubuntu:

```bash
python SIRST_main_all.py --dataset NUAA-SIRST-Old  --Inference_resize True  --split_method 50_50 --inference_path /media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_16_11_2023_00_20_25_Retrain
```


## Results and Trained Models
#### Qualitative Results

![outline](Qualitative_result.png)








