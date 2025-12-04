_base_ = [
    '_base_/models/faster-rcnn_r50_fpn_binary_explosive.py',
    '_base_/datasets/my2nd_coco_binary_exp.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' 