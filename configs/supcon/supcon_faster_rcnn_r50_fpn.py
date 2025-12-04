
# _base_ = './faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
_base_ = [
    '_base_/models/faster-rcnn_r50_fpn_specific_explosive.py',
    '_base_/datasets/my2nd_coco_specific_exp.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
    type='StandardRoIHeadSupCon',
    contrastive=True,
    lambda_contrast=0.1,
    bbox_head=dict(
        return_roi_feats=True
    )
)
    )
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' 
