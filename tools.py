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
        self.reduction = reduction

    def forward(self, inputs, targets):
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0-inputs)**self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**self.beta * (inputs)**self.alpha * torch.log(1.0 - inputs + 1e-14)
        loss = center_loss + other_loss

        if self.reduction == 'mean':
            batch_size = loss.size(0)
            loss = torch.sum(loss) / batch_size

        if self.reduction == 'sum':
            loss = torch.sum(loss) / batch_size

        return loss


class FocalLoss(nn.Module):
    def __init__(self,  weight=None, alpha=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_ind = (targets == 1.0).float()
        neg_ind = (targets != 1.0).float()
        pos_loss = -pos_ind * (1.0 - inputs)**self.alpha * torch.log(inputs + 1e-14)
        neg_loss = -neg_ind * (inputs)**self.alpha * torch.log(1.0 - inputs + 1e-14)
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            batch_size = loss.size(0)
            loss = torch.sum(loss) / batch_size

        if self.reduction == 'sum':
            loss = torch.sum(loss) / batch_size

        return loss


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


def generate_txtytwth(gt_label, w, h, s, gauss=False):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    box_w_s = box_w / s
    box_h_s = box_h / s

    if gauss:
        r = gaussian_radius([box_w_s, box_h_s])
        r = max(int(r), 1)
    else:
        r = None

    if box_w < 1e-4 or box_h < 1e-4:
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
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight, r


def gt_creator(input_size, stride, num_classes, label_lists=[], gauss=False):
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h, w = input_size
    
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, num_classes+4+1])

    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_cls = gt_label[-1]

            result = generate_txtytwth(gt_label, w, h, s, gauss=gauss)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight, r = result

                gt_tensor[batch_index, grid_y, grid_x, int(gt_cls)] = 1.0
                gt_tensor[batch_index, grid_y, grid_x, num_classes:num_classes + 4] = np.array([tx, ty, tw, th])
                gt_tensor[batch_index, grid_y, grid_x, num_classes + 4] = weight

                if gauss:
                    # get the x1x2y1y2 for the target
                    x1, y1, x2, y2 = gt_label[:-1]
                    x1s, x2s = int(x1 * ws), int(x2 * ws)
                    y1s, y2s = int(y1 * hs), int(y2 * hs)
                    # create the grid
                    grid_x_mat, grid_y_mat = np.meshgrid(np.arange(x1s, x2s), np.arange(y1s, y2s))
                    # create a Gauss Heatmap for the target
                    heatmap = np.exp(-((grid_x_mat - grid_x)**2 + (grid_y_mat - grid_y)**2) / (2*(r/3)**2))
                    p = gt_tensor[batch_index, y1s:y2s, x1s:x2s, int(gt_cls)]
                    gt_tensor[    batch_index, y1s:y2s, x1s:x2s, int(gt_cls)] = np.maximum(heatmap, p)
                
    gt_tensor = gt_tensor.reshape(batch_size, -1, num_classes+4+1)

    return gt_tensor


def loss(pred_cls, pred_txty, pred_twth, label, num_classes):
    # create loss_f
    cls_loss_function = HeatmapLoss(reduction='mean') # FocalLoss(reduction='mean')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')

    # groundtruth    
    gt_cls = label[:, :, :num_classes].float()
    gt_txty = label[:, :, num_classes:num_classes+2].float()
    gt_twth = label[:, :, num_classes+2:num_classes+4].float()
    gt_box_scale_weight = label[:, :, -1]

    # objectness loss
    batch_size = pred_cls.size(0)
    cls_loss = cls_loss_function(pred_cls, gt_cls)
        
    # box loss
    txty_loss = torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txty), 2) * gt_box_scale_weight) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(pred_twth, gt_twth), 2) * gt_box_scale_weight) / batch_size

    # total loss
    total_loss = cls_loss + txty_loss + twth_loss

    return cls_loss, txty_loss, twth_loss, total_loss


if __name__ == "__main__":
    pass