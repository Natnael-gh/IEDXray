_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

data_root = r"C:\Users\Natnael\Desktop\Datasets\2ndExperiment\complete_annotations\coco/"
# class_name =  ('Laptop', 'Mobile', 'Pager', 'Walkie-Talkie')
class_name = (  'Explosive',
                'Battery',
                'Modified laptop',
                'Modified parts',
                'Modified Mobile phone',
                'Modified Pager',
                'Modified Walkie Talkie',
                'Laptop',
                'Pager',
                'Mobile Phone',
                'Walkie-Talkie')

num_classes = len(class_name)
metainfo = dict(classes=class_name)

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

max_epoch = 30

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)
