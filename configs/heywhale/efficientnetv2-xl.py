_base_ = [
    '../efficientnet_v2/efficientnetv2-s_8xb32_in1k-384px.py',
]

load_from = None
resume = False

# dataset settings
batch_size = 4
num_workers = 0
num_classes = 5
classes = ['A', 'B', 'C', 'D', 'E']

# train cfg
max_epochs = 100
val_interval = 1
log_interval = 10
ckpt_interval = 1

lr = 0.1
momentum = 0.9
weight_decay = 0.0001

# model setting
model = dict(
    backbone=dict(in_channels=4, arch='xl'),
    head=dict(num_classes=num_classes),
    data_preprocessor=dict(
        mean=[0.485, 0.456, 0.406, 0.449],
        std=[0.229, 0.224, 0.225, 0.226],
        num_classes=num_classes))

train_pipeline = [
    dict(type='LoadNpyFromFile'),
    dict(type='EfficientNetRandomCrop', scale=256, crop_padding=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadNpyFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=224, crop_padding=0),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    dataset=dict(
        pipeline=train_pipeline,
        ann_file='',
        classes=classes,
        extensions=['.npy'],
    ))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    dataset=dict(
        pipeline=test_pipeline,
        ann_file='',
        classes=classes,
        extensions=['.npy'],
    ))
test_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        pipeline=test_pipeline,
        ann_file='',
    ))

_base_.val_evaluator['topk'] = 1
_base_.optim_wrapper['optimizer']['lr'] = lr
_base_.optim_wrapper['optimizer']['momentum'] = momentum
_base_.optim_wrapper['optimizer']['weight_decay'] = weight_decay

_base_.train_cfg['max_epochs'] = max_epochs
_base_.train_cfg['val_interval'] = val_interval

_base_.default_hooks['logger']['interval'] = log_interval
_base_.default_hooks['checkpoint']['interval'] = ckpt_interval
_base_.default_hooks['checkpoint']['save_best'] = 'accuracy/top1'
