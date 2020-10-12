# -*- coding: utf-8 -*-

import torch
import torch.jit


def main():

    # Read model.
    model_path = 'model\\ft_ResNet50\\'
    script_path = model_path + 'net_last.script'

    model = torch.jit.load(script_path)
    model = model.eval()

    input = torch.rand((2, 3, 256, 128))

    with torch.no_grad():
        output = model(input)
        print(output[0].size())
        print(output[0])
        print(output[1].size())
        print(output[1])


if __name__ == '__main__':

    print('Start...')
    main()
    print('End...')
