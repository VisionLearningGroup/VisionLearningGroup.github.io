## Moment Matching for Multi-Source Domain Adaptation
<img src='https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/blob/master/M3SDA/imgs/overview.png'>
PyTorch implementation for **Domain agnostic learning with disentangled representations** (ICCV2019 Oral). This repository contains some code from [Maximum Classifier Discrepancy for Domain Adaptation](https://github.com/mil-tokyo/MCD_DA). If you find this repository useful for you, please also consider cite the MCD paper!


The code has been tested on Python 3.6+PyTorch 0.3. To run the training and testing code, use the following script:

## Installation
- Install PyTorch (Works on Version 0.3) and dependencies from http://pytorch.org.
- Install Torch vision from the source.
- Install torchnet as follows
```
pip install git+https://github.com/pytorch/tnt.git@master
```

## DomainNet
The DomainNet dataset can be downloaded from the following link:
[http://ai.bu.edu/M3SDA/](http://ai.bu.edu/M3SDA/)

We are also organizing a TaskCV and VisDA chanllenge in conjunction with ICCV 2019, Seoul, Korea, based on this dataset. See more details with the following link:
[http://ai.bu.edu/visda-2019/](http://ai.bu.edu/visda-2019/)

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/1812.01754.pdf)
```
@article{peng2018moment,
        title={Moment Matching for Multi-Source Domain Adaptation},
        author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
        journal={arXiv preprint arXiv:1812.01754},
        year={2018}
        }
```
             
