import torch
import numpy as np
import torchvision


class Evaluator:
    def __init__(self):
        # loss: [total loss, loss_cls, loss_box_reg, loss_obj, loss_rpn_box_reg]
        self.loss = torch.empty((0, 5))

    @staticmethod
    def binary_pred(pred, cutoff):
        cp_pred = pred.clone()
        cp_pred[cp_pred < cutoff] = 0
        cp_pred[cp_pred >= cutoff] = 1
        # print(pred[1200:1400], cp_pred[1200:1400])
        # print(torch.max(pred))
        return cp_pred

    @staticmethod
    def confusion_matrix(label, pred):
        # print(label[:100] - pred[:100])
        true_neg = (label + pred == 0).sum().item()
        true_pos = (label + pred == 2).sum().item()
        false_neg = (label - pred == 1).sum().item()
        false_pos = (label - pred == -1).sum().item()
        return np.asarray([true_neg, true_pos, false_neg, false_pos])

    def reset_loss(self):
        self.loss = torch.empty((0, 5))

    def send_loss(self, losses):
        self.loss = torch.cat([self.loss, losses], dim=0)

    def all_in_one(self, net, data_loader):
        net.model.eval()
        # net.train()  # training mode to print losses !!!!!!!!!!
        n = len(data_loader)
        self.reset_loss()

        nms_cutoff = net.model.rpn.nms_thresh

        with torch.no_grad():
            conf_mat_sum = np.zeros(3)
            for images, targets in data_loader:
                with torch.cuda.amp.autocast():
                    # TODO find a way to output losses in eval mode
                    loss_and_predictions = net.model(images, targets)
                    loss_dict = loss_and_predictions[0]

                    # Process losses
                    # loss_mat = torch.tensor([i.item() for i in loss_dict.values()])
                    loss_mat = torch.tensor(list(map(lambda x: x.item(), loss_dict.values())))
                    total_losses = sum(loss_mat).reshape(1)
                    total_loss_mat = torch.cat([total_losses, loss_mat])
                    total_loss_mat = torch.unsqueeze(total_loss_mat, dim=0)
                    self.send_loss(total_loss_mat)

                    # Process predictions
                    predictions = loss_and_predictions[1]
                    for i in range(len(predictions)):
                        conf_mat_sum += self.bbox_pred_img(predictions[i]['boxes'], predictions[i]['scores'],
                                                           targets[i]['boxes'], nms_cutoff=nms_cutoff)

        sum_losses = torch.sum(self.loss, dim=0)
        avg_losses = sum_losses / n

        return np.asarray(avg_losses), conf_mat_sum.astype(np.int)

        # TODO: add in mAP under eval mode in future

    @staticmethod
    def bbox_pred_img(pred_bbox, pred_score, true_bbox, nms_cutoff=0.7, tp_cutoff=0.5):
        """
        Image level prediction.
        1. Get final prediction via NMS
        2. Compare final prediction with ground truth by iou
        return true_pos, false_pos, false_neg
        """
        final_pred_idx = torchvision.ops.nms(pred_bbox, pred_score, nms_cutoff)
        final_bbox = torch.index_select(pred_bbox, 0, final_pred_idx)
        iou_score = torchvision.ops.box_iou(final_bbox, true_bbox)  # TODO change out iou calculation here
        true_pos = torch.sum(iou_score > tp_cutoff).item()
        false_pos = final_bbox.size()[0] - true_pos
        if true_pos > true_bbox.size()[0]:  # handle double counting of true pos
            true_pos = true_bbox.shape[0]
        false_neg = true_bbox.size()[0] - true_pos
        conf_mat = np.asarray([true_pos, false_pos, false_neg])
        return conf_mat

    def bbox_pred(self, net, data_loader):
        net.model.eval()
        nms_cutoff = net.model.rpn.nms_thresh

        with torch.no_grad():
            conf_mat_sum = np.zeros(3)
            for images, targets in data_loader:
                predictions = net.model(images)
                for i in range(len(predictions)):
                    conf_mat_sum += self.bbox_pred_img(predictions[i]['boxes'], predictions[i]['scores'],
                                                       targets[i]['boxes'], nms_cutoff=nms_cutoff)
        return conf_mat_sum.astype(np.int)

    @staticmethod
    def metrics(conf_mat, beta=1):
        precision = conf_mat[0] / (conf_mat[0] + conf_mat[1] + 0.000001)
        recall = conf_mat[0] / (conf_mat[0] + conf_mat[2] + 0.000001)
        f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 0.000001)
        return np.array([precision, recall, f_score])
