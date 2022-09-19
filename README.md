# DropCov
Implementation of DropCov as described in DropCov: A Simple yet Effective Method for Improving Deep Architectures by  Qilong Wang, Mingze Gao, Zhaolin Zhang, Jiangtao Xie, Peihua Li, Qinghua Hu
<<<<<<< HEAD

# Requirements


# Main Results on ImageNet with Pretrained Models


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
python main.py --eval --resume <checkpoint> --model <model-name>--data-path <imagenet-path> 
```

For example, to evaluate the Dropcov method, run

```bash
python main.py --eval --resume --model .pth --data-path <imagenet-path>
```

giving
```bash
* Acc@1 Acc@5 loss
```

You can find all supported models in `models/registry.py.`

## Training

One can simply call the following script to run training process. Distributed training is recommended even on single GPU node. 

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --use_env main.py \
--model <model-name>
--data-path <imagenet-path>
--output_dir <output-path>
--dist-eval
```

# Citation

```
@article{,
  title={},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```

# Contributing


# Acknowledgement

=======
a
>>>>>>> e999273de2df91a7106c3e8ac5938eae96d8baf3
