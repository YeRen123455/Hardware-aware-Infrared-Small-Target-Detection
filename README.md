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
python test.py --base_size 256 --crop_size 256 --st_model [trained model path] --model_dir [model_dir] --dataset [dataset-name] --split_method 50_50 --model [model name] --backbone resnet_18  --deep_supervision True --test_batch_size 1 --mode TXT 
```


## Results and Trained Models
#### Qualitative Results

![outline](Qualitative_result.png)

#### Quantative Results 

on NUDT-SIRST

| Model         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| DNANet-VGG-10 | 85.23 | 96.95 | 6.782|
| DNANet-ResNet-10| 86.36 | 97.39 | 6.897 |
| DNANet-ResNet-18| 87.09 | 98.73 | 4.223 |
| DNANet-ResNet-18| 88.61 | 98.42 | 4.30 | [[Weights]](https://drive.google.com/file/d/1NDvjOiWecfWNPaO12KeIgiJMTKSFS6wj/view?usp=sharing) |
| DNANet-ResNet-34| 86.87 | 97.98 | 3.710 |


on NUAA-SIRST
| Model         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| DNANet-VGG-10 | 74.96 | 97.34 | 26.73 |
| DNANet-ResNet-10| 76.24 | 97.71 | 12.80 |
| DNANet-ResNet-18| 77.47 | 98.48 | 2.353 |
| DNANet-ResNet-18| 79.26 | 98.48 | 2.30 | [[Weights]](https://drive.google.com/file/d/1W0jFN9ZlaIdGFemYKi34tmJfGxjUGCRc/view?usp=sharing) |
| DNANet-ResNet-34| 77.54 | 98.10 | 2.510 |

on NUST-SIRST

| Model         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) ||
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| DNANet-ResNet-18| 46.73 | 81.29 | 33.87 | [[Weights]](https://drive.google.com/file/d/1TF0bZRMsGuKzMhlHKH1LygScBveMcCS2/view?usp=sharing) |

*This code is highly borrowed from [ACM](https://github.com/YimianDai/open-acm). Thanks to Yimian Dai.

*The overall repository style is highly borrowed from [PSA](https://github.com/jiwoon-ahn/psa). Thanks to jiwoon-ahn.

## Referrences

1. Dai Y, Wu Y, Zhou F, et al. Asymmetric contextual modulation for infrared small target detection[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2021: 950-959. [[code]](https://github.com/YimianDai/open-acm) 

2. Zhou Z, Siddiquee M M R, Tajbakhsh N, et al. Unet++: Redesigning skip connections to exploit multiscale features in image segmentation[J]. IEEE transactions on medical imaging, 2019, 39(6): 1856-1867. [[code]](https://github.com/MrGiovanni/UNetPlusPlus)

3. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778. [[code]](https://github.com/rwightman/pytorch-image-models)







