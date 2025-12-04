_base_ = [
    # '_base_/models/faster-rcnn_r50_fpn.py',
    '_base_/models/cascade-rcnn_r50_fpn_specific_exp.py',
    '_base_/datasets/my2nd_coco_specific_exp.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth' 