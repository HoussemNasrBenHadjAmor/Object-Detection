from models import *

def return_fasterrcnn_resnet50_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_resnet50_fpn_v2(num_classes, size):
    model = fasterrcnn_resnet50_fpn_v2.create_model(num_classes, size)
    return model

create_model = {
    'ssd200': return_fasterrcnn_resnet50_fpn,
    'ssd300': return_fasterrcnn_resnet50_fpn_v2
}