# -*- coding: utf-8 -*-

from model import ft_net
import torch


def convert():

    # Create model structure.
    model = ft_net(751)

    model_path = 'model\\ft_ResNet50\\'
    pth_path = model_path + 'net_last.pth'
    script_path = model_path + 'net_last.script'

    # Read model weight.
    model.load_state_dict(torch.load(pth_path))
    model.zero_grad()
    model = model.eval()

    print(model)

    # Save model.
    # torch.jit.script(model).save(script_path)


if __name__ == '__main__':

    print('Start...')
    convert()
    print('End...')
