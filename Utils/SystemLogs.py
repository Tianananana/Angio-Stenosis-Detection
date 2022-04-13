import socket
import os
import torch.cuda


def SystemLogs(device):
    print('OS process id ', os.getpid())
    print(socket.gethostname())
    print(device)

    if torch.cuda.is_available():
        print('cur cuda dev ', torch.cuda.current_device())
        print('total gpu available ', torch.cuda.device_count())
