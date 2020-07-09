# config.py
import os.path


# centernet config
voc_cfg = {
    'num_classes': 20,
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'min_dim': 512,
    'name': 'VOC',
}

coco_cfg = {
    'num_classes': 80,
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'min_dim': 512,
    'name': 'COCO',
}
