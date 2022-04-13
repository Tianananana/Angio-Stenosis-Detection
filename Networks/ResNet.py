""" Parts of the Encoder model """

from .Blocks import *


class ResNet(nn.Module):

    def __init__(self, mid_c, n_downsampling, n_bottleneck, fc_list, out_kernel, in_kernel, dropout, bias=True):
        super().__init__()

        self.op_c = mid_c * 2 ** n_downsampling

        model_conv = [nn.Conv2d(3, mid_c, kernel_size=out_kernel, stride=1, padding=out_kernel // 2, bias=bias),
                      nn.BatchNorm2d(mid_c),
                      nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model_conv += [
                ResBN(mid_c * mult, mid_c * mult * 2, k_size=in_kernel, stride=2, pad=in_kernel // 2, bias=bias)]

        for i in range(n_bottleneck):
            model_conv += [ResBN(self.op_c, self.op_c, k_size=in_kernel, stride=1, pad=in_kernel // 2, bias=bias)]

        model_conv += [nn.AdaptiveAvgPool2d(1)]

        self.model_conv = nn.Sequential(*model_conv)

        model_fc = []
        pre_neuron = self.op_c
        for n_neuron in fc_list:
            model_fc += [nn.Linear(pre_neuron, n_neuron),
                         nn.ReLU(True),
                         nn.Dropout(dropout)]
            pre_neuron = n_neuron

        model_fc += [nn.Linear(pre_neuron, 1),
                     nn.Sigmoid()]
        self.model_fc = nn.Sequential(*model_fc)

    def forward(self, x):
        feature = self.model_conv(x)
        # print('feature size before squeeze', feature.shape)
        feature = feature.view(-1, self.op_c)
        pred = self.model_fc(feature)
        return feature, pred
