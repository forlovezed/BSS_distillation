'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import os
import time

import torch
import torchvision
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as ds
import torchvision.models as models
from torchvision.models import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import attacks
from models.resnet_orig import *


def BN_version_fix(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    return net

dataset_name = "Cifar-10"
#res_folder = "./results/student/"
gpu_num = 0
NUM_EPOCH=10
model_file = "D:\BSS_distillation\\results\\students/cifar_resnet18_final.pth"
if not os.path.exists(model_file):
    print("Pretrained Model Un-founded !")
    exit()
#if not os.path.exists(res_folder):
#    os.mkdir(res_folder)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Testing on a GPU !")

print('==> Preparing test data..')
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914
                              , 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net = models.resnet50()
#net.load_state_dict(torch.load(model_file))
net = resnet18()
net.load_state_dict(torch.load(model_file))
#net.to(device)
total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0], data[1]
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Total num of samples is : %d" % total)
print("total num of Corrected Classification is : %d" % correct)
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))