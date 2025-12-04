_base_ = [
    '_base_\models\mask-rcnn_r50_fpn.py',
    '_base_/datasets/my_coco_maskrcnn_exp.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'