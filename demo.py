import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import BaseTransform, VOC_CLASSES, coco_class_index, coco_class_labels
from data import config
import numpy as np
import cv2
import tools
import time



def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Demo Detection')

    parser.add_argument('-v', '--version', default='centernet',
                        help='centernet.')
    parser.add_argument('-bk', '--backbone', default='r18',
                        help='r18, r34, r50, r101')
    parser.add_argument('--trained_model', default='weights/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('-size', '--input_size', default=512, type=int,
                        help='input_size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--conf_thresh', default=0.3, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.45, type=float,
                        help='NMS threshold')
    parser.add_argument('--path_to_img', default='data/demo/Images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/video/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_saveVid', default='data/video/result.avi',
                        type=str, help='The path to save the detection results video')
    parser.add_argument('-vs', '--visual_threshold', default=0.3,
                        type=float, help='visual threshold')
    parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                        help='use nms.')
    
    return parser.parse_args()
                    

def vis(img, bbox_pred, scores, cls_inds, class_color, thresh=0.3):
        
    for i, box in enumerate(bbox_pred):
        if scores[i] > thresh:
            cls_indx = cls_inds[i]
            cls_id = coco_class_index[int(cls_indx)]
            cls_name = coco_class_labels[cls_id]
            mess = '%s: %.3f' % (cls_name, scores[i])
            # bounding box
            xmin, ymin, xmax, ymax = box
            box_w = int(xmax - xmin)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmin+box_w*0.55), int(ymin)), class_color[int(cls_indx)], -1)

            cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img


def detect(net, device, transform, thresh, mode='image', path_to_img=None, path_to_vid=None, path_to_save=None):
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
    # ------------------------- Camera ----------------------------
    # I'm not sure whether this 'camera' mode works ...
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            cv2.imshow('current frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            detections = net(x)      # forward pass
            torch.cuda.synchronize()
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            scale = np.array([[frame.shape[1], frame.shape[0],
                                frame.shape[1], frame.shape[0]]])
            bbox_pred, scores, cls_inds = detections
            # map the boxes to origin image scale
            bbox_pred *= scale

            frame_processed = vis(frame, bbox_pred, scores, cls_inds, class_color, setup, thresh=thresh)
            cv2.imshow('detection result', frame_processed)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for file in os.listdir(path_to_img):
            img = cv2.imread(path_to_img + '/' + file, cv2.IMREAD_COLOR)
            x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            detections = net(x)      # forward pass
            torch.cuda.synchronize()
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            scale = np.array([[img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]]])
            bbox_pred, scores, cls_inds = detections
            # map the boxes to origin image scale
            bbox_pred *= scale

            img_processed = vis(img, bbox_pred, scores, cls_inds, class_color=class_color, setup=setup, thresh=thresh)
            cv2.imshow('detection result', img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output000.avi',fourcc, 40.0, (1280,720))
        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                t0 = time.time()
                x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)

                torch.cuda.synchronize()
                t0 = time.time()
                detections = net(x)      # forward pass
                torch.cuda.synchronize()
                t1 = time.time()
                print("detection time used ", t1-t0, "s")
                # scale each detection back up to the image
                scale = np.array([[frame.shape[1], frame.shape[0],
                                    frame.shape[1], frame.shape[0]]])
                bbox_pred, scores, cls_inds = detections
                # map the boxes to origin image scale
                bbox_pred *= scale
                
                frame_processed = vis(frame, bbox_pred, scores, cls_inds, class_color=class_color, setup=setup, thresh=thresh)
                out.write(frame_processed)
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    args = parse_args()

    # use cuda
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = [args.input_size, args.input_size]

    # load net
    if args.version == 'centernet':
        from models.centernet import CenterNet
        net = CenterNet(device, 
                        input_size=input_size, 
                        num_classes=80, 
                        backbone=args.backbone,
                        conf_thresh=args.conf_thresh, 
                        nms_thresh=args.nms_thresh, 
                        use_nms=args.use_nms)

    net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    net.to(device).eval()
    print('Finished loading model!')

    # run
    if args.mode == 'camera':
        detect(net, device, BaseTransform(net.input_size), 
                    thresh=args.visual_threshold, mode=args.mode)
    elif args.mode == 'image':
        detect(net, device, BaseTransform(net.input_size), 
                    thresh=args.visual_threshold, mode=args.mode, path_to_img=args.path_to_img)
    elif args.mode == 'video':
        detect(net, device, BaseTransform(net.input_size),
                    thresh=args.visual_threshold, mode=args.mode, path_to_vid=args.path_to_vid, path_to_save=args.path_to_saveVid)


if __name__ == '__main__':
    run()
