from __future__ import division

from utils.cocoapi_evaluator import COCOAPIEvaluator
from data import *
from utils.augmentations import SSDAugmentation
from data.cocodataset import *
import tools

import os
import random
import argparse
import time
import math
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser(description='CenterNet Detection')
parser.add_argument('-v', '--version', default='centernet',
                    help='centernet')
parser.add_argument('-t', '--testset', action='store_true', default=False,
                    help='COCO_val, COCO_test-dev dataset')
parser.add_argument('--dataset_root', default='/home/k303/object-detection/dataset/COCO/', 
                    help='Location of VOC root directory')
parser.add_argument('--trained_model', default='weights_yolo_v2/coco/', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--num_classes', default=80, type=int, 
                    help='The number of dataset classes')
parser.add_argument('--n_cpu', default=8, type=int, 
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode where only one image is trained')


args = parser.parse_args()
data_dir = args.dataset_root

def test(model, device):
    global cfg
    if args.testset:
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=cfg['min_dim'],
                        device=device,
                        testset=True,
                        transform=BaseTransform(cfg['min_dim'], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=cfg['min_dim'],
                        device=device,
                        testset=False,
                        transform=BaseTransform(cfg['min_dim'], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
                        )

    # COCO evaluation
    ap50_95, ap50 = evaluator.evaluate(model)
    print('ap50 : ', ap50)
    print('ap50_95 : ', ap50_95)


if __name__ == '__main__':
    global cfg

    cfg = coco_cfg
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if args.version == 'centernet':
        from models.centernet import CenterNet
        model = CenterNet(device, input_size=cfg['min_dim'], num_classes=args.num_classes)
    
    else:
        print('Unknown Version !!!')
        exit()

    # load model
    model.load_state_dict(torch.load(args.trained_model, map_location=device))
    model.eval().to(device)
    print('Finished loading model!')

    test(model, device)
