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
parser.add_argument('-size', '--input_size', default=512, type=float,
                    help='input size')
parser.add_argument('--trained_model', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.3, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                    help='use nms.')
parser.add_argument('--num_classes', default=80, type=int, 
                    help='The number of dataset classes')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')


args = parser.parse_args()
data_dir = coco_root

def test(model, device, input_size):
    if args.testset:
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=input_size,
                        device=device,
                        testset=True,
                        transform=BaseTransform(input_size)
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=input_size,
                        device=device,
                        testset=False,
                        transform=BaseTransform(input_size)
                        )

    # COCO evaluation
    ap50_95, ap50 = evaluator.evaluate(model)
    print('ap50 : ', ap50)
    print('ap50_95 : ', ap50_95)


if __name__ == '__main__':

    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_classes = 80
    input_size = [args.input_size, args.input_size]
    # load net
    if args.version == 'centernet':
        from models.centernet import CenterNet
        net = CenterNet(device, 
                        input_size=input_size, 
                        num_classes=num_classes, 
                        conf_thresh=args.conf_thresh, 
                        nms_thresh=args.nms_thresh, 
                        use_nms=args.use_nms)
    
    else:
        print('Unknown Version !!!')
        exit()

    # load model
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.eval().to(device)
    print('Finished loading model!')

    test(net, device, input_size)
