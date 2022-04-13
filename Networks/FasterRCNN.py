import torch.nn as nn

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from Code.FasterRCNN.Networks import ModelConfig
from Code.FasterRCNN.Networks.rpn2 import RPN2
from Code.FasterRCNN.Networks.roi_heads2 import RoIHead2
from Code.FasterRCNN.Utils.Utils import eager_outputs


## PARAMS for FasterRCNN ##
anchor_generator = AnchorGenerator(sizes=ModelConfig.scale,
                                   aspect_ratios=ModelConfig.aspect_ratios)
# RPN #
rpn_fg_iou_thresh = ModelConfig.rpn_fg_iou_thresh
rpn_bg_iou_thresh = ModelConfig.rpn_bg_iou_thresh
rpn_batch_size_per_image = ModelConfig.rpn_batch_size_per_image
rpn_positive_fraction = ModelConfig.rpn_positive_fraction
rpn_nms_thresh = ModelConfig.rpn_nms_thresh
rpn_score_thresh = ModelConfig.rpn_score_thresh

rpn_pre_nms_top_n_train = ModelConfig.rpn_pre_nms_top_n_train
rpn_pre_nms_top_n_test = ModelConfig.rpn_pre_nms_top_n_test
rpn_post_nms_top_n_train = ModelConfig.rpn_post_nms_top_n_train
rpn_post_nms_top_n_test = ModelConfig.rpn_post_nms_top_n_test

rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

# ROI Head #
out_channels = ModelConfig.out_channels

num_classes = ModelConfig.num_classes
representation_size = ModelConfig.representation_size


box_roi_pool = MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=12,
    sampling_ratio=2)

resolution = box_roi_pool.output_size[0]

box_head = TwoMLPHead(
    out_channels * resolution ** 2,
    representation_size)

box_predictor = FastRCNNPredictor(
    representation_size,
    num_classes)

box_fg_iou_thresh = ModelConfig.box_fg_iou_thresh
box_bg_iou_thresh = ModelConfig.box_bg_iou_thresh
box_batch_size_per_image = ModelConfig.box_batch_size_per_image
box_positive_fraction = ModelConfig.box_positive_fraction
bbox_reg_weights = ModelConfig.bbox_reg_weights
box_score_thresh = ModelConfig.box_score_thresh
box_nms_thresh = ModelConfig.box_nms_thresh
box_detections_per_img = ModelConfig.box_detections_per_img


class Model(nn.Module):
    def __init__(self, n=2, nms_thresh=0.7, new_model=True):
        """
        :param n: no. classes (including background). Default: 2
        :param nms_thresh: threshold for nms suppression
        :param pretrained: pretrained on Imagenet/coco dataset
        """
        super().__init__()
        self.n = n
        # LW's resnet vs resnet18
        self.backbone = ModelConfig.backbone
        self.backbone.out_channels = out_channels

        self.anchors = anchor_generator
        self.model = FasterRCNN(self.backbone,
                                num_classes=self.n,
                                rpn_anchor_generator=self.anchors,
                                box_roi_pool=box_roi_pool)

        # self.rpn_head = self.model.rpn.head
        # set nms threshold value
        self.model.rpn.nms_thresh = nms_thresh

        if new_model:
            # TODO Change out rpn, roi_heads, and eager_outputs
            FasterRCNN.eager_outputs = eager_outputs  # TODO!!! eager_outputs changed

            # TODO Put default and changable rpn params here

            self.model.rpn = RPN2(
                self.anchors, self.model.rpn.head,
                rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                rpn_batch_size_per_image, rpn_positive_fraction,
                rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
                score_thresh=rpn_score_thresh)

            # TODO Put default and changable roi params here
            self.model.roi_heads = RoIHead2(
                box_roi_pool, box_head, box_predictor,
                box_fg_iou_thresh, box_bg_iou_thresh,
                box_batch_size_per_image, box_positive_fraction,
                bbox_reg_weights,
                box_score_thresh, box_nms_thresh, box_detections_per_img)
