# DropCov
Implementation of DropCov as described in DropCov: A Simple yet Effective Method for Improving Deep Architectures by  Qilong Wang, Mingze Gao, Zhaolin Zhang, Jiangtao Xie, Peihua Li, Qinghua Hu

# Requirements


# Main Results on ImageNet with Pretrained Models
it 

| name               | acc@1 | #params | FLOPs | url                                                          |
| ------------------ | ----- | ------- | ----- | ------------------------------------------------------------ |
|     |   | M     | G  | [github](https://github.com/.pth) |

# Usage

## Install
First, clone the repo and install requirements:

```bash
git clone https://github.com/mingzeG/DropCov.git
pip install -r requirements.txt
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. 
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), 
and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation

To evaluate a pre-trained model on ImageNet val with a single GPU run:

```bash
CUDA_VISIBLE_DEVICES={device_ids}  python  -u main.py  -e -a {model_name} --resume {checkpoint-path} {imagenet-path}
```

For example, to evaluate the Dropcov method, run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python  -u main.py  -e -a resnet18_ACD --resume ./r18_64_acd_best.pth.tar ./dataset/ILSVRC2012
```

giving
```bash
* Acc@1 73.5 Acc@5 91.2
```

You can find all supported models in `models/registry.py.`

## Training

#### Train with ResNet

You can run the `main.py` to train or evaluate as follow:

```
CUDA_VISIBLE_DEVICES={device_ids} python -u main.py -a {model_name} --epochs {epoch_num} --b {batch_size} --lr_mode {the schedule of learning rate decline} {imagenet-path}
```
For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python  -u main.py  -a resnet18_ACD --epochs 100 --b 256 --lr_mode LRnorm  ./dataset/ILSVRC2012
```

# Citation

```
@article{,
  title={A Simple yet Effective Method for Improving Deep Architectures},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```

# Contributing


# Acknowledgement


