"""
Image-level analysis of model.
Plot predicted bbox into images for visualization.
"""

import os
import sys
import torch

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import Config
from Utils import PostProcess, collate, mydevice
from DataSet import DataSet
from Preprocessing import plot_bbox


def main():
    # load trained model onto same network architecture
    print('Loading: ', weight_path)
    if not os.path.exists(weight_path):
        sys.exit('No trained model exists.')

    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['net_state_dict'])
    model.to(device)

    train_set = DataSet(data_dir, fname[0], transform=False)
    test_set = DataSet(data_dir, fname[1], transform=False)
    valid_set = DataSet(data_dir, fname[2], transform=False)

    # TODO: copy all batch size out into config file
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    loader_all = [train_loader, valid_loader, test_loader]
    loader_name = ['train/', 'valid/', 'test/']
    id_all = [train_set.id_, valid_set.id_, test_set.id_]

    model.model.eval()
    with torch.no_grad():
        # TODO automate new directories for train/test/valid
        for k in range(len(loader_name)):
            print(f'checking {loader_name[k]} ...')
            sub_path = op_path + loader_name[k]
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
            id_ = id_all[k]
            batch_no = 0

            true_pos = 0
            false_neg = 0
            false_pos = 0

            for images, targets in tqdm(loader_all[k]):  # predict train images for now

                losses, predictions = model.model(images, targets)
                for i in range(len(predictions)):
                    # per image
                    image_id = id_[batch_no * batch_size + i, :]
                    image_id = '_'.join([str(i.item()) for i in image_id])

                    # TODO: use original (unmasked) image
                    img = plt.imread(
                        f"/home/chentyt/Documents/4tb/Annotation/v3_combined_annotate_24-01-2021/{image_id}.png")
                    img = np.moveaxis(img, -1, 0)[:1, :, :] * 255
                    # img2 = images[i].cpu().numpy()
                    box_pred = predictions[i]['boxes']
                    box_score = predictions[i]['scores']
                    box_targ = targets[i]['boxes']

                    ## POST-PROCESSING ##
                    all_idx = torch.arange(box_pred.size()[0])

                    # 1. Intersection over area
                    IoA_score = PostProcess.IoA(box_pred.cpu())
                    # print(IoA_score)

                    # 2. Select indexes below cutoff to delete
                    idx_to_del = (IoA_score < IoA_cutoff).nonzero()[:, 1]
                    idx_to_del = torch.unique(idx_to_del)

                    # 3. Delete indexes
                    filter_idx = torch.from_numpy(np.setdiff1d(all_idx.numpy(), idx_to_del.numpy()))
                    filter_box_pred = torch.index_select(box_pred.cpu(), 0, filter_idx)

                    # use nms supression
                    # filter_box_idx = torchvision.ops.nms(boxes=box_pred, scores=box_score, iou_threshold=nms_cutoff)
                    # filter_box_pred = torch.index_select(box_pred, 0, filter_box_idx)
                    # filter_box_score = torch.index_select(box_score, 0, filter_box_idx)
                    # filter_box_idx = torchvision.ops.nms(boxes=filter_box_pred, scores=filter_box_score, iou_threshold=nms_cutoff)
                    # filter_box_pred = torch.index_select(filter_box_pred, 0, filter_box_idx)

                    # output lesion statistics
                    # if (len(filter_box_pred) != 0) or (len(box_score) != 0):
                    prediction_num = filter_box_pred.size()[0]
                    target_num = box_targ.size()[0]
                    if (prediction_num != 0) or (target_num != 0):
                        # wrongly predicted
                        if target_num == 0:
                            false_pos += prediction_num
                        # undetected stenosis
                        if prediction_num == 0:
                            false_neg += target_num
                        else:
                            eval_IoA = PostProcess.IoA2(filter_box_pred, box_targ)
                            pred_IoA_idx, targ_IoA_idx = torch.where(eval_IoA > IoA_cutoff)
                            pred_IoA_idx, targ_IoA_idx = pred_IoA_idx.unique(), targ_IoA_idx.unique()
                            # print(pred_IoA_idx, targ_IoA_idx)
                            # print(f'pred_IoA:{len(pred_IoA_idx)} \t targ_IoA:{len(targ_IoA_idx)}')
                            curr_tp = len(pred_IoA_idx)
                            true_pos += curr_tp

                            false_pos += (prediction_num - curr_tp)
                            false_neg += (target_num - curr_tp)
                            pass

                    filter_box_pred = filter_box_pred.cpu().numpy()
                    box_targ = box_targ.cpu().numpy()

                    plotted_img = plot_bbox(np.empty((0, 4)), img)
                    plotted_targ = plot_bbox(box_targ, img, color=[0, 0, 255])
                    plotted_pred = plot_bbox(filter_box_pred, img, color=[255, 0, 0])
                    plotted_overlap = plot_bbox(filter_box_pred, plotted_targ, overlap=True)
                    plotted = np.vstack([np.expand_dims(plotted_img, 0),
                                         np.expand_dims(plotted_targ, 0),
                                         np.expand_dims(plotted_pred, 0),
                                         np.expand_dims(plotted_overlap, 0)])

                    f = plt.figure(figsize=(15, 5))
                    title = ['Image', 'Label', 'Predict', 'Overlap']
                    for j in range(4):
                        f.add_subplot(1, 4, j + 1)
                        plt.imshow(plotted[j, :, :, :])
                        plt.title(title[j])
                        # plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])

                    if save:
                        plt.savefig(sub_path + f'{image_id}.png')
                    if show:
                        plt.show()

                    plt.clf()
                    plt.close()

                batch_no += 1

            print(f"True positive: {true_pos} \t False positive: {false_pos} \t False negative: {false_neg}")
            print(
                f"Total correctly identified lesions: {true_pos}/{true_pos + false_neg} "
                f"\t Recall: {true_pos / (true_pos + false_neg)}")
            print(
                f"Wrong lesion localization: {false_pos}/{true_pos + false_pos} "
                f"\t Precision: {true_pos / (true_pos + false_pos)}")

            # TODO
            # add output statistics on lesion after nms: [tp, fp, fn].


if __name__ == '__main__':
    data_dir = Config.data_dir
    fname = Config.fname
    output_dir = Config.output_dir

    device = mydevice
    # TODO INPUT MODEL ID
    # model_id = Config.model_id
    model_id = "2022-03-30_Resnet_simple_F1Val_rpnNms0.3_boxBatchSize:64,lr0.0001,clsAlpha('None',),n_channel32," \
               "kSize5anchorScale(128,256,512)run3 "
    name = 'IoA'
    op_path = f'/home/chentyt/Documents/4tb/Tiana/P100ObjDet/BboxChecker/{model_id}/{name}/'
    if not os.path.exists(op_path):
        os.makedirs(op_path)

    weight_path = f'{output_dir}{model_id}_weight'
    batch_size = Config.batch_size

    model = Config.model

    show = False
    save = False
    nms_cutoff = 0.5
    IoA_cutoff = 0.7
    main()
