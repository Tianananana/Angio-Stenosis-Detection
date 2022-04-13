# make this into a singleton
import torch

select_GPU = "cuda:0" #or cuda:1

class DeviceSingleton:
    class __Device:
        def __init__(self, ):
            use_cuda = torch.cuda.is_available()
            self.value = torch.device(select_GPU if use_cuda else "cpu")

    # global instance in class
    instance = None

    def __init__(self):
        if not DeviceSingleton.instance:
            DeviceSingleton.instance = DeviceSingleton.__Device()
        return  # does nothing if already constructed

    def get_device(self):
        return self.instance.value


# ================================================

# make globally accessible
mydevice = DeviceSingleton().get_device()


def verify_device(specified_device):
    # use_cuda = torch.cuda.is_available()
    target = torch.device(specified_device)

    if mydevice != target:
        print('WARNING: requested device unavailable', mydevice)

    # this is to test if loading to GPU will throw an error
    if mydevice == 'cuda':
        a = torch.tensor([1])
        a = a.to('cuda')
