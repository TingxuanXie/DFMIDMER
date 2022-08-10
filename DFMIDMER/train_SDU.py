
from __future__ import print_function
import numpy as np

import os
import sys
import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
from torchvision import transforms

from dataset.load_data import load_data
from torch.utils.data import DataLoader
from losses import DivRegLoss
from torch.autograd import Variable
from models import BiCnet_TKS
import transforms.temporal_transforms as TT
from models.KLloss import kl_loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, criterion, criterion_div, optimizer):
    """
    one epoch training
    """
    model.train()

    train_total = 0
    train_correct = 0
    loss_all = 0.0
    margin = 0

    for idx, (inputs, label, index) in enumerate(train_loader):

        inputs = inputs.to(device)
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(device)

        # ===================forward=====================
        # output, _, _ = model(inputs)
        # output, _ = model(inputs)
        output, xh, xl, masks = model(inputs)


        loss_cross = criterion(output, label).float()
        loss_kl = kl_loss(xh, xl).float()
        loss_div = criterion_div(masks).float()
        # print('loss_kl', loss_kl)
        loss = margin + loss_cross - loss_kl + loss_div

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_all += loss.item()

        _, predicted = torch.max(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum()
    train_acc = 100. * train_correct / train_total
    return loss_all, train_acc


def main():
    # load data
    classes = 6

    flow_root = r'/data/SDU/'
    train_file = r'/data/txt_file/train_label.txt'
    test_file = r'/data/txt_file/train_label.txt'
    save_path = r'save_model/SDU/'


    tranform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    temporal_transform = TT.TemporalRandomCrop(size=6, stride=4)

    train_set = load_data(file=train_file, root=flow_root, transform=tranform, temporal_transform=temporal_transform)
    test_set = load_data(file=test_file, root=flow_root, transform=tranform, temporal_transform=temporal_transform)


    train_loader = DataLoader(
        dataset=train_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=False)


    # num of samples
    n_data = len(train_set)
    print('number of samples: {}'.format(n_data))

    # set the model
    model = BiCnet_TKS(num_classes=classes)
    model = model.to(device)
    print('model\n')
    print(model)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    # set the criterion
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_div = DivRegLoss()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # train
    print('start train......')
    epochs = 300
    best_acc = 0
    best_epoch = 0
    for epoch in range(0, epochs + 1):
        loss_all, train_acc = train(train_loader, model, criterion, criterion_div, optimizer)

        print('epoch:%d, loss_all: %.03f  | Acc: %.3f%%' % (epoch + 1, loss_all, train_acc))
        if epoch > 1:
        # if (epoch + 1) % 3 == 0:
            model.eval()
            print('waiting test')
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for idx, (inputs, label, index) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    label = torch.tensor(label, dtype=torch.long)
                    label = label.to(device)
                    output, _, _, _ = model(inputs)
                    # output, _ = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    test_total += label.size(0)
                    test_correct += (predicted == label).sum()
            acc = 100. * test_correct / test_total
            print('the accuracy of test %.3f%%' % (acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                save_checkpoint(state=model.state_dict(), epoch=epoch, save_path=save_path)
    print('best_epoch {},best_acc {}'.format(best_epoch, best_acc))
    print('finished train')


def save_checkpoint(state, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename ='{}_bicnet_checkpoint.pth.tar'.format(epoch)
    file_path=os.path.join(save_path, filename)
    torch.save(state, file_path)


if __name__ == '__main__':
    main()