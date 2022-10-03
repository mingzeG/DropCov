# DropCov
Implementation of DropCov as described in DropCov: A Simple yet Effective Method for Improving Deep Architectures by  Qilong Wang, Mingze Gao, Zhaolin Zhang, Jiangtao Xie, Peihua Li, Qinghua Hu

![Poster](figures/method.png)

# Main Results on ImageNet with Pretrained Models


|Method           | acc@1(%) | #params.(M) | FLOPs(G) | url                                                          |
| ------------------ | ----- | ------- | ----- | ------------------------------------------------------------ |
| ResNet-34   |  74.19 |  21.8   |  3.66   |               |
| ResNet-50   |  76.02 |  25.6   |   3.86  |               |
| ResNet-101   |  77.67 |    44.6 | 7.57    |               |
| ResNet-34+DropCov(Ours)   | 76.81  |  29.6   | 5.56    |               |
| ResNet-50+DropCov(Ours)   | 78.19  |   32.0  |  6.19   |               |
| ResNet-101+DropCov(Ours)    |  79.51 |    51.0 |   9.90  |               |
| DeiT-S   |  79.8 |  22.1   |   4.6  |               |
| Swin-T   |  81.2 |   28.3  |     4.5|               |
| T2T-ViT-14   |  81.5 |    21.5 |   5.2  |               |
| DeiT-S+DropCov(Ours)   | 82.2  |   25.5  |     |               |
| Swin-T-S+DropCov(Ours)  |  82.5 |   31.6  |   6.0  |               |
| T2T-ViT-14-S+DropCov(Ours)   | 82.7  |  24.9   |    5.4 |               |
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

To evaluate a pre-trained model on ImageNet val with GPUs run:

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

## Training

#### Train with ResNet

You can run the `main.py` to train as follow:

```
CUDA_VISIBLE_DEVICES={device_ids} python -u main.py -a {model_name} --epochs {epoch_num} --b {batch_size} --lr_mode {the schedule of learning rate decline} {imagenet-path}
```
For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python  -u main.py  -a resnet18_ACD --epochs 100 --b 256 --lr_mode LRnorm  ./dataset/ILSVRC2012
```
#### Train with Swin
`Swin-T`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py  --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 
```
#### Train with Deit and T2T
`Deit-S`:
```bash
sh ./scripts/train_Deit_drop_Small.sh
```
# Citation

```
@article{,
  title={A Simple yet Effective Method for Improving Deep Architectures},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```

# Acknowledgement


Our code are built following 
[GCP_Optimization](https://github.com/ZhangLi-CS/GCP_Optimization),
[DeiT](https://github.com/facebookresearch/deit),
[Swin Transformer](https://github.com/microsoft/Swin-Transformer)
, thanks for their excellent work


