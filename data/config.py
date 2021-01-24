# config.py
import os.path


# centernet config
train_cfg = {
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'min_dim': [512, 512],
}
