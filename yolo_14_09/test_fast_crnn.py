#
# from torch import nn
# import torch.nn.functional as F
#from torchvision.ops import MultiScaleRoIAlign
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,)

# num_classes = 2
#
# in_features = model.roi_heads.box_predictor.cls_score.in_features
#
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
