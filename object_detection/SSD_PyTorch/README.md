# SSD: Single Shot MultiBox Object Detector, in PyTorch

<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>


## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
- Then download the dataset below.
# Summary
## Intro
- SSD is designed for object detection in real-time and it also eliminate the need for the region proposal network. To recover the drop in accuracy, SSD applies a few improvements including multi-scale features and default boxes which allow SSD to match the Faster R-CNN’s accuracy using lower resolution images, which further pushes the speed higher. 
###  *Comparison 
<img src= "https://github.com/Shrayansh19/SSD-Model-Zoo/blob/main/doc/1_rqGEyJKbKv3ecmjaMSiEtA.png">

### The SSD object detection composes of 2 parts:
- Extract feature maps
- Apply convolution filters to detect objects
<img src= "https://github.com/Shrayansh19/SSD-Model-Zoo/blob/main/doc/1_aex5im2aYcsk4RVKUD4zeg.jpeg">

SSD uses VGG16 to extract feature maps. Then it detects objects using the Conv4_3 layer. 

## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).


### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
```

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">

## Performance

#### VOC2007 Test

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |

##### FPS
**GTX 1060:** ~45.45 FPS

## Contributor

* [**Shrayansh Jakar**](https://github.com/Shrayansh19)

## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [Cornell Paper](https://arxiv.org/abs/1512.02325).
- COCO: Common Objects in Context. http://mscoco.org/dataset/
#detections-leaderboard (2016) [Online; accessed 25-July-2016].
- He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR.
(2016)
