# dataset settings
dataset_type = 'CocoDataset'
# data_root = r'C:\Users\Natnael\Desktop\Complete Dataset With Captions\captions\split_2_specific_electronic_device\coco/'
# data_root = r"C:\Users\Natnael\Desktop\Datasets\Task 3 Specific Explosive Detection/"
# data_root = r"C:\Users\Natnael\Desktop\Datasets\2ndExperiment\device_detection\coco/"
# data_root = r"C:\Users\Natnael\Desktop\Datasets\split_2_electronic_explosive_binary/"
data_root = r"C:\Users\Natnael\Desktop\Datasets\2ndExperiment\complete_annotations\coco/"
# data_root = r"C:\Users\Natnael\Desktop\Datasets\New test natnael\split_2_specific_electronic_device/"
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

# backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
# train_dataloader = dict(
#     # metainfo = dict(classes=('Laptop', 'Pager', 'Mobile Phone', 'Walkie-Talkie')),
#     batch_size=16,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args=backend_args))
# val_dataloader = dict(
#     # metainfo = dict(classes=('Laptop', 'Pager', 'Mobile Phone', 'Walkie-Talkie')),
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/instances_val2017.json',
#         data_prefix=dict(img='val2017/'),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))
# test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root+'annotations/instances_val2017.json',
#     metric='bbox',
#     format_only=False,
#     backend_args=backend_args)
# test_evaluator = val_evaluator











backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        # MMDetection-style meta info
        metainfo = dict(
            classes=(
                'Explosive',
                'Battery',
                'Modified laptop',
                'Modified parts',
                'Modified Mobile phone',
                'Modified Pager',
                'Modified Walkie Talkie',
                'Normal Laptop',
                'Normal Pager',
                'Normal Mobile Phone',
                'Normal Walkie-Talkie',
            )
        ),
        # metainfo = dict(classes=('Laptop', 'Mobile', 'Pager', 'Walkie-Talkie')),
        # metainfo = dict(classes=('explosive')),
        # metainfo= dict(classes=('IED Explosive', 'Laptop Explosive', 'Pager Explosive', 'Mobile Phone Explosive', 'Walkie-Talkie Explosive'), palette=[(220, 220, 60), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]),

        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        # MMDetection-style meta info
        metainfo = dict(
            classes=(
                'Explosive',
                'Battery',
                'Modified laptop',
                'Modified parts',
                'Modified Mobile phone',
                'Modified Pager',
                'Modified Walkie Talkie',
                'Normal Laptop',
                'Normal Pager',
                'Normal Mobile Phone',
                'Normal Walkie-Talkie',
            )
        ),
        # metainfo = dict(classes=('Laptop', 'Mobile', 'Pager', 'Walkie-Talkie')),
        # metainfo = dict(classes=('explosive')),
        # metainfo= dict(classes=('IED Explosive', 'Laptop Explosive', 'Pager Explosive', 'Mobile Phone Explosive', 'Walkie-Talkie Explosive'), palette=[(220, 220, 60), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]),

        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator














# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
