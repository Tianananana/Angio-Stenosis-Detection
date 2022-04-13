import sys
import torch
import numpy as np
from DataSet import DataSet
import Config
from SupervisedTrainer import SupervisedTrainer
from Evaluator import Evaluator
from Model_IO import train_model
from Utils import mydevice
from Utils import SystemLogs

# IO params
seed = Config.seed
data_dir = Config.data_dir
log_dir = Config.log_dir
model_dir = Config.model_dir

fname = Config.fname

# Model params
device = mydevice
model = Config.model
model.to(device)

epoch_no = Config.epoch_no
patience = Config.patience
lr = Config.lr
batch_size = Config.batch_size


def manual_seed():
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(torch.backends.cudnn.deterministic)
    # print(torch.backends.cudnn.benchmark)
    np.random.seed(seed)


def main():
    torch.cuda.empty_cache()
    SystemLogs(device)  # print the hostname, pid, device etc

    manual_seed()

    trainer = SupervisedTrainer(net=model, lr=lr)
    evaluator = Evaluator()

    train_set = DataSet(data_dir, fname[0])
    test_set = DataSet(data_dir, fname[1], transform=False)
    valid_set = DataSet(data_dir, fname[2], transform=False)

    train_model(start_epoch=0, end_epoch=epoch_no,
                batch_size=batch_size, patience=patience, criteria_min=-sys.maxsize,
                trainer=trainer, evaluator=evaluator,
                train_set=train_set, valid_set=valid_set, test_set=test_set,
                model_path=model_dir, log_path=log_dir)


if __name__ == '__main__':
    main()
