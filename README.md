# X-modaler (COS-Net only)
This is a **COS-Net-only** distribution of [X-modaler](https://xmodaler.readthedocs.io/en/latest/), a versatile and high-performance codebase for cross-modal analytics. Everything unrelated to image captioning (video captioning, vision-language pre-training, visual question answering, visual commonsense reasoning and cross-modal retrieval) and every image-captioning model other than COS-Net (Attention, LSTM-A3, Up-Down, GCN-LSTM, Transformer, Meshed-Memory, X-LAN, TDEN, SCD-Net) have been removed.

The original X-modaler paper can be found [here](https://arxiv.org/pdf/2108.08217.pdf).

<p align="center">
  <img src="images/task.jpg" width="800"/>
</p>

## Installation
See [installation instructions](https://xmodaler.readthedocs.io/en/latest/tutorials/installation.html).

### Requiremenets
* Linux or macOS with Python ≥ 3.6
* PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this
* fvcore
* jsonlines
* pycocotools

## Getting Started
See [Getting Started with X-modaler](https://xmodaler.readthedocs.io/en/latest/tutorials/getting_started.html)

### Training & Evaluation in Command Line

We provide a script in "train_net.py" to train the COS-Net configs provided here. You may want to use it as a reference to write your own training script.

First setup the datasets following [datasets](xmodaler/datasets/README.md) and the COS-Net [data preparation](configs/image_caption/cosnet/README.md), then run:
```
# Teacher Force
python train_net.py --num-gpus 4 \
 	--config-file configs/image_caption/cosnet/cosnet.yaml

# Reinforcement Learning
python train_net.py --num-gpus 4 \
 	--config-file configs/image_caption/cosnet/cosnet_rl.yaml
```

## Model Zoo and Baselines

<table>
  <tr>
    <td colspan="4" align="center"><font size=3><b>Image Captioning</b></font></td>
  </tr>
  <tr>
    <td>COS-Net</td>
    <td> Comprehending and Ordering Semantics for Image Captioning </td>
    <td>CVPR</td>
    <td>2022</td>
  </tr>
</table>

Models and results for COS-Net can be downloaded from the links in [configs/image_caption/cosnet/README.md](configs/image_caption/cosnet/README.md).

## License
X-modaler is released under the [Apache License, Version 2.0](LICENSE).

## Citing X-modaler
If you use X-modaler in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{Xmodaler2021,
  author =       {Yehao Li, Yingwei Pan, Jingwen Chen, Ting Yao, and Tao Mei},
  title =        {X-modaler: A Versatile and High-performance Codebase for Cross-modal Analytics},
  booktitle =    {Proceedings of the 29th ACM international conference on Multimedia},
  year =         {2021}
}
```

## Citing COS-Net
```BibTeX
@inproceedings{cosnet2022cvpr,
  title={Comprehending and Ordering Semantics for Image Captioning},
  author={Li, Yehao and Pan, Yingwei and Yao, Ting and Mei, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
