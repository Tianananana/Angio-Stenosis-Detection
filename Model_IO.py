import time
import torch
import numpy as np

from Utils import collate
from torch.utils.data import DataLoader

torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)


def train_model(start_epoch, end_epoch, batch_size, patience, criteria_min, trainer, evaluator,
                train_set, valid_set, test_set, model_path, log_path):
    p_count = 0
    min_epoch = None
    criteria_name = "Val F1"  # TODO

    # use_cuda = torch.cuda.is_available()
    # kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    for i in range(start_epoch, end_epoch):
        start_time = time.time()

        train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate)
        valid_loader = DataLoader(valid_set, batch_size, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(test_set, batch_size, shuffle=False, collate_fn=collate)

        # print('started training')
        trainer.one_epoch(train_loader)
        # print('finished training')

        # generate new train loader for evaluation purposes
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, collate_fn=collate)

        train_loss_mat, train_conf_mat = evaluator.all_in_one(trainer.net, train_loader)
        valid_loss_mat, valid_conf_mat = evaluator.all_in_one(trainer.net, valid_loader)
        test_loss_mat, test_conf_mat = evaluator.all_in_one(trainer.net, test_loader)

        train_metric = evaluator.metrics(train_conf_mat)
        valid_metric = evaluator.metrics(valid_conf_mat)
        test_metric = evaluator.metrics(test_conf_mat)

        interval = (time.time() - start_time) / 60

        message = f'epoch {i} \t time: {interval:.2f} \t ' \
                  f'train: loss {train_loss_mat}, classification {train_conf_mat}, metric {train_metric} \t' \
                  f'valid: loss {valid_loss_mat}, classification {valid_conf_mat}, metric {valid_metric} \t' \
                  f'test: loss {test_loss_mat}, classification {test_conf_mat}, metric {test_metric}'

        print(message)
        with open(log_path, 'a') as log:
            log.write(message + '\n')

        # set criteria to save weights
        criteria = valid_metric[2]  # TODO

        if criteria > criteria_min:  # TODO
            min_epoch = i
            p_count = 0
            criteria_min = criteria

            print(f'saving weights for new {criteria_name} {criteria} ... weight updated.')
            torch.save({'epoch': i,
                        'net_state_dict': trainer.net.state_dict(),
                        'optimizer_state_dict': trainer.opt.state_dict(),
                        'criteria_min': criteria_min}, model_path)

        else:
            p_count += 1

        # early stopping if valid_mean_acc plateau for 100 epochs
        if p_count > patience:
            print(f'{criteria_name} did not increase for {patience} epochs consequently.')
            break
    # print(f"Final lr: {trainer.scheduler.get_last_lr()}")

    with open(log_path, 'a') as log:
        log.write(f'{criteria_name} epoch: {min_epoch}' + '\n')

    print('training ended.')
    return
