_base_ = ['../resnet/resnet50_8xb32_in1k.py']

load_from = None
resume = False

num_classes = 5
classes = ['A', 'B', 'C', 'D', 'E']

batch_size = 32
num_workers = 0
persistent_workers = num_workers > 0
max_epochs = 100
val_interval = 1
log_interval = 10
ckpt_interval = 1

lr = 0.1
momentum = 0.9
weight_decay = 0.0001

# model settings
model = dict(
    backbone=dict(in_channels=4),
    head=dict(num_classes=num_classes),
    data_preprocessor=dict(
        mean=[0.485, 0.456, 0.406, 0.449], std=[0.229, 0.224, 0.225, 0.226]))

train_pipeline = [
    dict(type='LoadNpyFromFile'),
    dict(type='RandomResizedCropC4', crop_ratio_range=(0.9, 1.0), scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadNpyFromFile'),
    dict(type='ResizeEdgeC4', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

# dataset settings
_base_.data_preprocessor['num_classes'] = num_classes

_base_.train_dataloader['batch_size'] = batch_size
_base_.train_dataloader['num_workers'] = num_workers
_base_.train_dataloader['persistent_workers'] = persistent_workers
_base_.train_dataloader['dataset']['classes'] = classes
_base_.train_dataloader['dataset']['ann_file'] = ''
_base_.train_dataloader['dataset']['extensions'] = ['.npy']
_base_.train_dataloader['dataset']['pipeline'] = train_pipeline

_base_.val_dataloader['batch_size'] = batch_size
_base_.val_dataloader['num_workers'] = num_workers
_base_.val_dataloader['persistent_workers'] = persistent_workers
_base_.val_dataloader['dataset']['classes'] = classes
_base_.val_dataloader['dataset']['ann_file'] = ''
_base_.val_dataloader['dataset']['extensions'] = ['.npy']
_base_.val_dataloader['dataset']['pipeline'] = test_pipeline

# train and eval setting
# _base_.val_evaluator['topk'] = 1
val_evaluator = dict(type='BCELossMetric', topk=1)
test_evaluator = val_evaluator

_base_.optim_wrapper['optimizer']['lr'] = lr
_base_.optim_wrapper['optimizer']['momentum'] = momentum
_base_.optim_wrapper['optimizer']['weight_decay'] = weight_decay

_base_.param_scheduler['milestones'] = [30, 60, 90]

_base_.train_cfg['max_epochs'] = max_epochs
_base_.train_cfg['val_interval'] = val_interval

_base_.default_hooks['logger']['interval'] = log_interval
_base_.default_hooks['checkpoint']['interval'] = ckpt_interval
# _base_.default_hooks['checkpoint']['save_best'] = 'accuracy/top1'
_base_.default_hooks['checkpoint']['save_best'] = 'loss/bce_loss'
