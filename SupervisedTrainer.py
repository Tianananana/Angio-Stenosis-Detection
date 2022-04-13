import torch
import torch.optim as optim
from Utils import check_grads


class SupervisedTrainer:

    def __init__(self, net, lr):
        self.net = net
        self.net.model.train()
        self.opt = optim.Adam(self.net.parameters(), lr)
        # added scheduler to adjust lr per epoch
        # self.scheduler = scheduler.LambdaLR(self.opt, lr_lambda=lambda epoch: epoch * 0.97)

    def one_epoch(self, data_loader):
        # set faster RCNN to training mode
        self.net.model.train()

        for images, targets in data_loader:
            # print(images.shape, targets.shape)

            with torch.cuda.amp.autocast():
                loss_dict = self.net.model(images, targets)
                # print(loss_dict)
                values = list(loss_dict.values())
                values[0] = values[0]  # * 10
                values[1] = values[1]  # * 10
                losses = sum(loss for loss in values)

                self.opt.zero_grad()
                losses.backward()
                self.opt.step()
            # self.scheduler.step()

        # only check gradients of last batch
        # check_grads(self.net.model.backbone, "backbone",
        #             grad_path="/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Model/")
