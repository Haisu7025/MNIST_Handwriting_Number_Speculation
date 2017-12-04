# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.autograd import Variable
from models import Net
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

checkpoint_path = 'checkpoints/mnist_state_epoch11.pth'


def prep_data(target_path='test/0.jpg'):
    raw_img = Image.open(target_path)
    raw_img = raw_img.convert('L')
    raw_img = raw_img.resize((28, 28))
    arr = np.array(raw_img)
    for i in range(28):
        for j in range(28):
            arr[i][j] = 255 - arr[i][j]
    raw_img = Image.fromarray(arr, mode='L')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tar_img = transform(raw_img)
    tar_img_a = tar_img.numpy()
    tar_img_a = tar_img[np.newaxis, :]
    tar_img = torch.Tensor(tar_img_a)
    return tar_img


def main():
    model = Net()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    tar = prep_data()
    output = model(Variable(tar))
    res = output.cpu().data.numpy()
    res_ = np.squeeze(res)
    num = np.argwhere(res_ == np.max(res_))
    print(int(num))


if __name__ == '__main__':
    main()
