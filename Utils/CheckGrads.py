import torch
import warnings

def check_grads(net, net_name, grad_path='./', save=True):
    """
    Computes gradients of current weights.
    Checks that gradients of weights within the network are properly updated after backpropagation.
    Copied from Nic's code.
    :param net: torch.nn.Module
    :param net_name: str
    """
    message = f"check grads for {net_name}"
    print(message, end='\t')
    # print(f'{message:#^80}')
    # check gradients for None
    print('check None gradients', end='\t')
    none_flag = False
    for name, param in net.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                # print(f'gradient of None found for name: {name}, of value {param.grad}')
                warnings.warn(f'gradient of None found for name: {name}, of value {param.grad}')
                none_flag = True

    # if none_flag:
    #     # print('exiting, a None gradient is found')
    #     warnings.warn('exiting, a None gradient is found')
    #     quit()
    # else:
    #     print('"None" gradients checked, no errors - no gradients are None')

    # Check the gradients for Nan
    print('check NaN gradients', end='\t')
    none_flag = False
    for name, param in net.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if torch.any(torch.isnan(param.grad)):
                    # print(f"gradient of NaN found for name: {name}, of value {param.grad}")
                    warnings.warn(f"gradient of NaN found for name: {name}, of value {param.grad}")
                    none_flag = True
    # if none_flag:
    #     # print('exiting, a NaN gradient is found')
    #     warnings.warn('exiting, a NaN gradient is found')
    #     quit()

    # else:
        # print('"Nan gradients checked, no errors - no gradients are Nan"')

    # check huge gradients
    print('check huge gradients', end='\t')
    none_flag = False
    for name, param in net.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if torch.any(abs(param.grad) > 1e12):
                    # print(f"huge gradient of more than 1e12 found for name: {name}, of value {param.grad}")
                    warnings.warn(f"huge gradient of more than 1e12 found for name: {name}, of value {param.grad}")
    # if none_flag:
    #     # print('exiting, erroneous huge gradients are found')
    #     warnings.warn('exiting, erroneous huge gradients are found')
    #     quit()

    # else:
    #     print('huge gradients chcked, no errors - no gradients are too huge')

    # check small gradients
    print('check small gradients', end='\t')
    none_flag = False
    for name, param in net.named_parameters():

        if param.requires_grad:
            if param.grad is not None:
                if torch.all(abs(param.grad) < 1e-12):
                    # print(f"small gradient of less than 1e-12 found for name: {name}, of value {param.grad}")
                    warnings.warn(f"small gradient of less than 1e-12 found for name: {name}, of value {param.grad}")
                    none_flag = True
    # if none_flag:
    #     # print('exiting, erroneous small gradients is found')
    #     warnings.warn('erroneous small gradients is found')
    #     quit()
    # else:
    #     print('small gradients checked, no errors - no gradients are small')

    # print(f'{"end of check_grads:":#^80}\n')
    print(f'end of check_grads.', end='\t')

    # output gradients for selected layer
    layer = 0
    count = 0

    grad_path = grad_path + f'gradients_{layer}.log'
    for name, param in net.named_parameters():
        if layer == count:
            mean = torch.mean(param.grad).item()
            with open(grad_path, 'a') as log:
                if save:
                    log.write(f'Mean gradients for layer {layer}: {mean} + \n')
                print(f'Mean gradients for layer {layer}: {mean}')
            break
        count += 1

    pass

