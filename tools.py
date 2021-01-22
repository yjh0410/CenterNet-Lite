import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F



class HeatmapLoss(nn.Module):
    def __init__(self,  weight=None, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0-inputs)**self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**self.beta * (inputs)**self.alpha * torch.log(1.0 - inputs + 1e-14)

        return center_loss + other_loss


def gaussian_radius(det_size, min_overlap=0.7):
    box_h, box_h  = det_size
    a1 = 1
    b1 = (box_h + box_h)
    c1 = box_h * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2 #(2*a1)

    a2 = 4
    b2 = 2 * (box_h + box_h)
    c2 = (1 - min_overlap) * box_h * box_h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2 #(2*a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_h + box_h)
    c3 = (min_overlap - 1) * box_h * box_h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2 #(2*a3)

    return min(r1, r2, r3)


def generate_txtytwth(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    box_w_s = box_w / s
    box_h_s = box_h / s

    r = gaussian_radius([box_w_s, box_h_s])
    sigma_w = sigma_h = r / 3
    # sigma_w = box_w_s / 6
    # sigma_h = box_h_s / 6

    if box_w < 1e-28 or box_h < 1e-28:
        # print('A dirty data !!!')
        return False    

    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # compute the (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w_s)
    th = np.log(box_h_s)
    weight = 1.0 # 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight, sigma_w, sigma_h


def gt_creator(input_size, stride, num_classes, label_lists=[]):
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h, w = input_size
    
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, num_classes+4+1])

    grid_x_mat, grid_y_mat = np.meshgrid(np.arange(ws), np.arange(hs))
    # generate gt whose style is yolo-v1
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_cls = gt_label[-1]
            result = generate_txtytwth(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight, sigma_w, sigma_h = result

                gt_tensor[batch_index, grid_y, grid_x, int(gt_cls)] = 1.0
                gt_tensor[batch_index, grid_y, grid_x, num_classes:num_classes + 4] = np.array([tx, ty, tw, th])
                gt_tensor[batch_index, grid_y, grid_x, num_classes + 4] = weight

                # create Gauss heatmap
                heatmap = np.exp(- (grid_x_mat - grid_x) ** 2 / (2*sigma_w**2) - (grid_y_mat - grid_y)**2 / (2*sigma_h**2))
                pre_v = gt_tensor[batch_index, :, :, int(gt_cls)]
                gt_tensor[batch_index, :, :, int(gt_cls)] = np.maximum(heatmap, pre_v)

                # create Gauss heatmap
                # for i in range(grid_x - 3*int(sigma_w), grid_x + 3*int(sigma_w) + 1):
                #     for j in range(grid_y - 3*int(sigma_h), grid_y + 3*int(sigma_h) + 1):
                #         if i < ws and j < hs:
                #             v = np.exp(- (i - grid_x)**2 / (2*sigma_w**2) - (j - grid_y)**2 / (2*sigma_h**2))
                #             pre_v = gt_tensor[batch_index, j, i, int(gt_cls)]
                #             gt_tensor[batch_index, j, i, int(gt_cls)] = max(v, pre_v)

    gt_tensor = gt_tensor.reshape(batch_size, -1, num_classes+4+1)

    return gt_tensor


def loss(pred_cls, pred_txty, pred_twth, label, num_classes):
    # create loss_f
    cls_loss_function = HeatmapLoss()
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')

    # groundtruth    
    gt_cls = label[:, :, :num_classes].float()
    gt_txtytwth = label[:, :, num_classes:-1].float()
    gt_box_scale_weight = label[:, :, -1]

    # objectness loss
    batch_size = pred_cls.size(0)
    cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls)) / batch_size
        
    # box loss
    txty_loss = torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight) / batch_size

    # total loss
    total_loss = cls_loss + txty_loss + twth_loss

    return cls_loss, txty_loss, twth_loss, total_loss


if __name__ == "__main__":
    pass