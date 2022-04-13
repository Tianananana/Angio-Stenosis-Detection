"""
Config settings for Faster RCNN Components:
1. Anchor Generator
2. Backbone
3. RPN
4. ROI
"""
import torch.nn as nn
from Code.FasterRCNN.Networks.ResNet import ResNet
# from Code.FasterRCNN.Config import roi_bbatch, n_channel_2d, k_size
import torchvision.models as models
import torch
from torchinfo import summary

# Anchor Generator #
# scale = ((10, 20, 40, 80),)
scale = ((128, 256, 512),)  # default scaling
aspect_ratios = ((0.5, 1.0, 2.0),)

# Backbone #
n_channel_2d = 32  # default: 32
n_downsampling = 3
n_bottleneck = 1
out_kernel = 5  # default: 5
in_kernel = 5  # default: 5
drop_out = 0.5
fc_list = []
bias = True  # TODO

# Backbone
# default Resnet
# resnet = models.resnet18(pretrained=False)
# resnet.load_state_dict(torch.load('/home/chentyt/Documents/4tb/PretrainedWeights/resnet18.pth'))
# backbone = nn.Sequential(*list(resnet.children())[:-2])
# out_channels = 512 # look from model

# Liu Wei's Resnet (less params)
resnet_simple = ResNet(n_channel_2d, n_downsampling, n_bottleneck, fc_list, out_kernel, in_kernel, drop_out, bias=bias)
backbone = nn.Sequential(*list(resnet_simple.children())[0][:-1])
out_channels = n_channel_2d * 2 ** n_downsampling

# summary(backbone, (10, 3, 512, 512))
# summary(backbone)
# backbone_simple = nn.Sequential(*list(resnet_simple.children())[0][:-1])
# summary(backbone_simple)

# RPN #
rpn_fg_iou_thresh = 0.7
rpn_bg_iou_thresh = 0.3
rpn_batch_size_per_image = 256
rpn_positive_fraction = 0.5
rpn_nms_thresh = 0.7
rpn_score_thresh = 0.0

# TODO KEEP LESSER TOP K PREDICTIONS
rpn_pre_nms_top_n_train = 2000
rpn_pre_nms_top_n_test = 1000
rpn_post_nms_top_n_train = 2000
rpn_post_nms_top_n_test = 1000

# RoI Head #
num_classes = 2
representation_size = 1024

box_fg_iou_thresh = 0.5
box_bg_iou_thresh = 0.5
box_batch_size_per_image = 64  # TODO
box_positive_fraction = 0.5  # TODO
bbox_reg_weights = None
box_score_thresh = 0.05
box_nms_thresh = 0.5
box_detections_per_img = 100
