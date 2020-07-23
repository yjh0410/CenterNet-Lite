# CenterNet-Lite
A PyTorch version of CenterNet（objects as points）. I only support resnet18 version. No DLA or Hourglass version.

I have trained it on VOC0712 and COCO 2017. You can download them from BaiDuYunDisk：

Link：https://pan.baidu.com/s/170OYftGRVW-j5qAKYyHSQQ

Password：jz4q

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td></tr>

<tr><th align="right" bgcolor=#f8f8f8> (official) resnet18 + DCN </th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 28 </td><td bgcolor=white> 44.9 </td></tr>

<tr><th align="right" bgcolor=#f8f8f8> (Our) resnet18 + SPP </th><td bgcolor=white> COCO val </td><td bgcolor=white> 25.3 </td><td bgcolor=white> 44.8 </td></tr>

</table></tbody>

I'm still trying something new to make my CenterNet-Lite stronger.

## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

## Dataset
### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is ```data/VOCdevkit/VOC2007``` and ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### MSCOCO Dataset
I copy the download files from the following excellent project:
https://github.com/DeNA/PyTorch_YOLOv3

#### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


## Train
### VOC
```Shell
python train_voc.py --cuda
```

You can run ```python train_voc.py -h``` to check all optional argument.

### COCO
```Shell
python train_coco.py --cuda
```

## Test
### VOC
```Shell
python test_voc.py --cuda --trained_model [ Please input the path to model dir. ]
```

### COCO
```Shell
python test_coco.py --cuda --trained_model [ Please input the path to model dir. ]
```


## Evaluation
### VOC
```Shell
python eval_voc.py --cuda --train_model [ Please input the path to model dir. ]
```

### COCO
To run on COCO_val:

```Shell
python eval_coco.py --cuda --train_model [ Please input the path to model dir. ]
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):

```Shell
python eval_coco.py --cuda -t --train_model [ Please input the path to model dir. ]
```
You will get a .json file which can be evaluated on COCO test server.