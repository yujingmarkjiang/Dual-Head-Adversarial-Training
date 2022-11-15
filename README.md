# DH-AT: Dual Head Adversarial Training 
This is the code for the [IJCNN'21 paper](https://arxiv.org/pdf/2104.10377.pdf) "Dual Head Adversarial Training" by Yujing Jiang, Xingjun Ma, Sarah Monazam Erfani, and James Bailey.

The base architecture utilizes [SAT](https://github.com/MadryLab/cifar10_challenge), [TRADES](https://github.com/yaodongyu/TRADES), and [MART](https://github.com/YisenWang/MART).

A [pre-trained checkpoint](https://drive.google.com/file/d/1I4guLRhpa90IK7I8n7Uq4CK9X5IvV5UW/view?usp=sharing) using DH-AT with TRADES is available.

## Prerequisites
* Python (3.7.4)
* Pytorch (1.4.0)
* CUDA (with 2 GPUs)
* numpy

## Reference
For technical details and full experimental results, please check [the paper](https://arxiv.org/pdf/2104.10377.pdf).
```
@inproceedings{jiang2021dual,
  title={Dual Head Adversarial Training},
  author={Jiang, Yujing and Ma, Xingjun and Erfani, Sarah Monazam and Bailey, James},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```
