_base_ = [
    '_base_/models/faster-rcnn_r50_fpn_complete.py',
    '_base_/datasets/my2nd_coco_complete.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' 

#  python tools/train.py configs\my_2nd_fasterrcnn_device_detection.py --work-dir work_dirs\Faster_rcnn_2nd_device_det_Fasterrcnn