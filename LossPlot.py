"""
Extract loss plots from log file
"""
import matplotlib.pyplot as plt
import numpy as np
import Config


def main():

    train_loss_l = np.empty((0, 5))
    train_class_l = np.empty((0, 3))
    train_metric_l = np.empty((0, 3))

    valid_loss_l = np.empty((0, 5))
    valid_class_l = np.empty((0, 3))
    valid_metric_l = np.empty((0, 3))

    test_loss_l = np.empty((0, 5))
    test_class_l = np.empty((0, 3))
    test_metric_l = np.empty((0, 3))

    with open(file_path, 'r') as LOG:
        for line in LOG.readlines()[:-1]:
            if 'weight updated' in line:
                continue
            val = line.split('\t')
            # epoch = val[0].split(' ')[1]
            # epoch_list.append(int(epoch))
            train_val = val[2].split(':')[-1]
            valid_val = val[3].split(':')[-1]
            test_val = val[4].split(':')[-1]

            train_val = train_val.split(',')
            valid_val = valid_val.split(',')
            test_val = test_val.split(',')

            train_loss, train_class, train_metric = train_val[0], train_val[1], train_val[2]
            valid_loss, valid_class, valid_metric = valid_val[0], valid_val[1], valid_val[2]
            test_loss, test_class, test_metric = test_val[0], test_val[1], test_val[2]

            train_loss = train_loss.split('[')[-1][:-1]
            train_loss = np.fromstring(train_loss, sep=' ')
            train_class = train_class.split('[')[-1][:-1]
            train_class = np.fromstring(train_class, sep=' ')
            train_metric = train_metric.split('[')[-1][:-1]
            train_metric = np.fromstring(train_metric, sep=' ')

            valid_loss = valid_loss.split('[')[-1][:-1]
            valid_loss = np.fromstring(valid_loss, sep=' ')
            valid_class = valid_class.split('[')[-1][:-1]
            valid_class = np.fromstring(valid_class, sep=' ')
            valid_metric = valid_metric.split('[')[-1][:-1]
            valid_metric = np.fromstring(valid_metric, sep=' ')

            test_loss = test_loss.split('[')[-1][:-1]
            test_loss = np.fromstring(test_loss, sep=' ')
            test_class = test_class.split('[')[-1][:-1]
            test_class = np.fromstring(test_class, sep=' ')
            test_metric = test_metric.split('[')[-1][:-1]
            test_metric = np.fromstring(test_metric, sep=' ')

            train_loss_l = np.concatenate([train_loss_l, np.expand_dims(train_loss, axis=0)], axis=0)
            train_class_l = np.concatenate([train_class_l, np.expand_dims(train_class, axis=0)], axis=0)
            train_metric_l = np.concatenate([train_metric_l, np.expand_dims(train_metric, axis=0)], axis=0)

            valid_loss_l = np.concatenate([valid_loss_l, np.expand_dims(valid_loss, axis=0)], axis=0)
            valid_class_l = np.concatenate([valid_class_l, np.expand_dims(valid_class, axis=0)], axis=0)
            valid_metric_l = np.concatenate([valid_metric_l, np.expand_dims(valid_metric, axis=0)], axis=0)

            test_loss_l = np.concatenate([test_loss_l, np.expand_dims(test_loss, axis=0)], axis=0)
            test_class_l = np.concatenate([test_class_l, np.expand_dims(test_class, axis=0)], axis=0)
            test_metric_l = np.concatenate([test_metric_l, np.expand_dims(test_metric, axis=0)], axis=0)
            # print(val)
            # print(epoch)
            # print(train_val, valid_val, test_val)

    # validation min epoch
    min_epoch_valid = np.argmin(valid_loss_l[:, 0])
    # performance at min_epoch
    print(f"Min valid loss: {valid_loss_l[min_epoch_valid, 0]} at epoch {min_epoch_valid}")
    print(f"Valid metrics at epoch {min_epoch_valid}: {valid_metric_l[min_epoch_valid]}")
    print(f"Test metrics at epoch {min_epoch_valid}: {test_metric_l[min_epoch_valid]}")

    max_epoch_F1 = np.argmax(valid_metric_l[:, 2])
    print(f"Min F1 loss: {valid_metric_l[max_epoch_F1, 2]} at epoch {max_epoch_F1}")
    print(f"Valid metrics at epoch {max_epoch_F1}: {valid_metric_l[max_epoch_F1]}")
    print(f"Test metrics at epoch {max_epoch_F1}: {test_metric_l[max_epoch_F1]}")

    if LOG:
        with open(file_dir + 'models.log', 'a') as LOG:
            LOG.write(f"{model_id}: \t")
            LOG.write(f"Min Valid loss: epoch {min_epoch_valid} \t Loss: {valid_loss_l[min_epoch_valid, 0]} \t")
            LOG.write(f"Valid metrics: {valid_metric_l[min_epoch_valid]} \t")
            LOG.write(f"Test metrics: {test_metric_l[min_epoch_valid]} \t")

            LOG.write(f"Max Valid F1: epoch {max_epoch_F1} \t F1: {valid_metric_l[max_epoch_F1, 2]} \t")
            LOG.write(f"Valid metrics: {valid_metric_l[max_epoch_F1]} \t")
            LOG.write(f"Test metrics: {test_metric_l[max_epoch_F1]} \n")

    # PLOTTING #
    # 9*9 subplots
    train_c, valid_c, test_c = 'b', 'orange', 'r'
    fig, ax = plt.subplots(3, 3, figsize=[12, 9])

    # [0, 0]: Total Loss
    ax[0, 0].plot(train_loss_l[:, 0], color=train_c)
    ax[0, 0].plot(valid_loss_l[:, 0], color=valid_c)
    ax[0, 0].plot(test_loss_l[:, 0], color=test_c)
    ax[0, 0].set_title('Total Loss')

    # [0, 1]: R-CNN Loss Classifier
    ax[0, 1].plot(train_loss_l[:, 1], color=train_c)
    ax[0, 1].plot(valid_loss_l[:, 1], color=valid_c)
    ax[0, 1].plot(test_loss_l[:, 1], color=test_c)
    ax[0, 1].set_title('R-CNN Loss Classifier')

    # [0, 2]: R-CNN Loss Regression
    ax[0, 2].plot(train_loss_l[:, 2], color=train_c)
    ax[0, 2].plot(valid_loss_l[:, 2], color=valid_c)
    ax[0, 2].plot(test_loss_l[:, 2], color=test_c)
    ax[0, 2].set_title('R-CNN Loss Regression')

    # [1, 1]: RPN Loss Objectness
    ax[1, 1].plot(train_loss_l[:, 3], color=train_c)
    ax[1, 1].plot(valid_loss_l[:, 3], color=valid_c)
    ax[1, 1].plot(test_loss_l[:, 3], color=test_c)
    ax[1, 1].set_title('RPN Loss Objectness')

    # [1, 2]: RPN Loss Regression
    ax[1, 2].plot(train_loss_l[:, 4], color=train_c)
    ax[1, 2].plot(valid_loss_l[:, 4], color=valid_c)
    ax[1, 2].plot(test_loss_l[:, 4], color=test_c)
    ax[1, 2].set_title('RPN Loss Regression')

    # [2, 0]: Precision
    ax[2, 0].plot(train_metric_l[:, 0], color=train_c)
    ax[2, 0].plot(valid_metric_l[:, 0], color=valid_c)
    ax[2, 0].plot(test_metric_l[:, 0], color=test_c)
    ax[2, 0].set_title('Precision')
    ax[2, 0].set_ylim([0, 1.05])

    # [2, 1]: Recall
    ax[2, 1].plot(train_metric_l[:, 1], color=train_c)
    ax[2, 1].plot(valid_metric_l[:, 1], color=valid_c)
    ax[2, 1].plot(test_metric_l[:, 1], color=test_c)
    ax[2, 1].set_title('Recall')
    ax[2, 1].set_ylim([0, 1.05])

    # [2, 2]: F Score
    ax[2, 2].plot(train_metric_l[:, 2], color=train_c)
    ax[2, 2].plot(valid_metric_l[:, 2], color=valid_c)
    ax[2, 2].plot(test_metric_l[:, 2], color=test_c)
    ax[2, 2].set_title('F Score')
    ax[2, 2].set_ylim([0, 1.05])

    # plt.setp(ax[:, :], legend=['train', 'valid', 'test'])
    plt.setp(ax[:, :], xlabel='Epoch')
    plt.setp(ax[:2, :], ylabel='Loss')
    plt.setp(ax[2, :], ylabel='%')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.legend(['train', 'valid', 'test'], loc='upper right')
    plt.suptitle(model_id, size=16)

    if save:
        plt.savefig(op_dir + model_id + "_fig.png")

    if show:
        plt.show()


if __name__ == '__main__':
    show = True
    save = True
    log = True
    file_dir = "/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Model/"
    model_id = Config.model_id
    # model_id = "2022-03-31_Resnet_simple_F1Val_rpnNms0.3_boxBatchSize:64,lr1e-05,clsAlpha('None',),n_channel32,
    # kSize5anchorScale(128,256,512)"
    op_dir = "/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Model/"
    file_path = file_dir + model_id + '_log'
    main()
