_base_ = ['../resnet/resnet50_8xb32_in1k.py']

# load_from = None
load_from = 'resnet50_8xb32_in1k_20210831-ea4938fc.pth'
resume = False

num_classes = 2
classes = ['negative', 'positive']
img_size = 448

batch_size = 8
num_workers = 0
persistent_workers = num_workers > 0
max_epochs = 100
val_interval = 1
log_interval = 10
ckpt_interval = 1

lr = 0.01
momentum = 0.9
weight_decay = 0.0001

cur_label = 'A'
data_root = f'data/imagenet_bin/{cur_label}'
work_dir = f'work_dir/resnet50_{cur_label}'

# model settings
model = dict(head=dict(num_classes=num_classes))

# dataset settings
_base_.data_preprocessor['num_classes'] = num_classes

_base_.train_dataloader['batch_size'] = batch_size
_base_.train_dataloader['num_workers'] = num_workers
_base_.train_dataloader['persistent_workers'] = persistent_workers
_base_.train_dataloader['dataset']['classes'] = classes
_base_.train_dataloader['dataset']['data_root'] = data_root
_base_.train_dataloader['dataset']['ann_file'] = ''
_base_.train_dataloader['dataset']['pipeline'][1]['scale'] = img_size

_base_.val_dataloader['batch_size'] = batch_size
_base_.val_dataloader['num_workers'] = num_workers
_base_.val_dataloader['persistent_workers'] = persistent_workers
_base_.val_dataloader['dataset']['classes'] = classes
_base_.val_dataloader['dataset']['data_root'] = data_root
_base_.val_dataloader['dataset']['ann_file'] = ''
_base_.val_dataloader['dataset']['pipeline'][1]['scale'] = int(img_size *
                                                               1.143)
_base_.val_dataloader['dataset']['pipeline'][2]['crop_size'] = img_size

# train and eval setting
_base_.val_evaluator['topk'] = 1

_base_.optim_wrapper['optimizer']['lr'] = lr
_base_.optim_wrapper['optimizer']['momentum'] = momentum
_base_.optim_wrapper['optimizer']['weight_decay'] = weight_decay

# _base_.param_scheduler['milestones'] = [30, 60, 90]
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=0,
        end=max_epochs)
]

_base_.train_cfg['max_epochs'] = max_epochs
_base_.train_cfg['val_interval'] = val_interval

_base_.default_hooks['logger']['interval'] = log_interval
_base_.default_hooks['checkpoint']['interval'] = ckpt_interval
_base_.default_hooks['checkpoint']['save_best'] = 'accuracy/top1'
