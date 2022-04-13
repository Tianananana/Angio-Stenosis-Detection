import torch.nn as nn


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)


class ResBN(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride, pad, bias):
        super().__init__()
        self.k_size = k_size
        self.stride = stride
        self.pad = pad
        self.model1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_size, stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(out_c, out_c // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(True),
            nn.Conv2d(out_c // 2, out_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        # print('Down block...')
        # print('k_size {} stride {} pad {} input shape {}'.format(self.k_size, self.stride, self.pad, x.shape))
        x = self.model1(x)
        # print(torch.cuda.memory_summary(abbreviated=True))
        x = x + self.model2(x)
        x = nn.functional.relu(x)  # no necessary, always greater than 0
        # print('ResBNDown', torch.max(out))
        return x
