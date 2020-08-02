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
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import attacks
from models.resnet_orig import *


dataset_name = "Cifar-10"
res_folder = "./results/students/"
gpu_num = 0
NUM_EPOCH = 20

if not os.path.exists(res_folder):
    os.mkdir(res_folder)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Training on a GPU !")

print('==> Preparing data..')
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = resnet18(pretrained=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
if use_cuda:
    torch.cuda.set_device(gpu_num)
    net.cuda()
    cudnn.benchmark = True

for epoch in range(NUM_EPOCH):
    running_loss = 0.0
    epoch_start_time = time.time()
    print('\npre-train students Epoch: %d' % epoch)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 40 == 39:  # print every 80 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 40))
            running_loss = 0.0
    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f ' % (running_loss / (i + 1)))
PATH = res_folder + "cifar_resnet18_final.pth"
torch.save(net.state_dict(), PATH)
#Net = net()
#Net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))