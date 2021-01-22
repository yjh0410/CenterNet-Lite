import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import VOC_ROOT, VOC_CLASSES
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import config
import numpy as np
import cv2
import tools
import time


parser = argparse.ArgumentParser(description='CenterNet Detection')
parser.add_argument('-v', '--version', default='centernet',
                    help='centernet')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-size', '--input_size', default=512, type=float,
                    help='input size')
parser.add_argument('--trained_model', default='weight/voc/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.3, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('-vs', '--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                    help='use nms.')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()


def test_net(net, device, testset, transform, thresh):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img_raw = testset.pull_image(index)

        img_tensor, _, h, w = testset.pull_item(index)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        t0 = time.time()
        bboxes, scores, cls_inds = net(img_tensor)
        print("detection time used ", time.time() - t0, "s")
        # map the boxes to origin image scale
        scale = np.array([[w, h, w, h]])
        bboxes *= scale

        CLASSES = VOC_CLASSES
        class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(20)]
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img_raw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
                cv2.rectangle(img_raw, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_color[int(cls_indx)], -1)
                mess = '%s' % (CLASSES[int(cls_indx)])
                cv2.putText(img_raw, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('detection', img_raw)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)


def test():
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = [args.input_size, args.input_size]
    num_classes = len(VOC_CLASSES)
    testset = VOCDetection(root=VOC_ROOT, 
                            img_size=None, 
                            transform=BaseTransform(input_size),
                            image_sets=[('2007', 'test')])

    # load net
    if args.version == 'centernet':
        from models.centernet import CenterNet
        net = CenterNet(device, 
                        input_size=input_size, 
                        num_classes=num_classes, 
                        conf_thresh=args.conf_thresh, 
                        nms_thresh=args.nms_thresh, 
                        use_nms=args.use_nms)
    
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(net, device, testset,
             BaseTransform(net.input_size),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test()