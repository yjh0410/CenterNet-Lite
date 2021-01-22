import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import 
from data import config, BaseTransform, VOCDetection, VOC_ROOT, VOC_CLASSES
from data import coco_root, coco_class_index, coco_class_labels
import numpy as np
import cv2
import time


parser = argparse.ArgumentParser(description='CenterNet Detection')
parser.add_argument('-v', '--version', default='centernet',
                    help='centernet')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='we use VOC-test or COCO-val to test.')
parser.add_argument('-size', '--input_size', default=512, type=float,
                    help='input size')
parser.add_argument('--trained_model', default='weights/',
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

args = parser.parse_args()

def test_net(net, device, testset, transform, thresh, mode='voc'):
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
    num_images = len(testset)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        if args.dataset == 'COCO':
            img, _ = testset.pull_image(index)
            img_tensor, _, h, w, offset, scale = testset.pull_item(index)
        elif args.dataset == 'VOC':
            img = testset.pull_image(index)
            img_tensor, _, h, w, offset, scale = testset.pull_item(index)

        x = img_tensor.unsqueeze(0).to(device)

        t0 = time.time()
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        # scale each detection back up to the image
        max_line = max(h, w)
        # map the boxes to input image with zero padding
        bboxes *= max_line
        # map to the image without zero padding
        bboxes -= (offset * max_line)

        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_color[int(cls_indx)], -1)
                cls_id = coco_class_index[int(cls_indx)]
                cls_name = coco_class_labels[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('detection', img)
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

    # load net
    num_classes = 80
    input_size = [args.input_size, args.input_size]

    if args.dataset == 'COCO':
        testset = COCODataset(
                    data_dir=args.dataset_root,
                    json_file='instances_val2017.json',
                    name='val2017',
                    img_size=input_size[0])

    elif args.dataset == 'VOC':
        testset = VOCDetection(VOC_ROOT, img_size=None, image_sets=[('2007', 'test')])

    # load net
    if args.version == 'centernet':
        from models.centernet import CenterNet
        net = CenterNet(device, 
                        input_size=input_size, 
                        num_classes=num_classes, 
                        conf_thresh=args.conf_thresh, 
                        nms_thresh=args.nms_thresh, 
                        use_nms=args.use_nms)

    net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(net, device, testset,
             BaseTransform(net.input_size),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test()