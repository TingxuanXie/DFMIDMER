from models.SingleResNet18_id import *
from torchvision import transforms
import transforms.temporal_transforms as TT
from dataset.load_data_id import load_data_id
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():

    classes = 73
    train_file = r'/txt_file/train_label_id.txt'
    test_file = r'/txt_file/test_label_id.txt'
    root = r'/data/SDU/'

    spatial_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    temporal_transform = TT.TemporalRandomCrop(size=6, stride=4)

    train_set = load_data_id(file=train_file, root=root,
                          spatial_transform=spatial_transform,
                          temporal_transform=temporal_transform)
    test_set = load_data_id(file=test_file, root=root,
                             spatial_transform=spatial_transform,
                             temporal_transform=temporal_transform)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=8,
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
    model = SingleResNet18_id(num_classes=classes)
    model = model.to(device)
    print('model\n')
    print(model)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # set the criterion
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    cudnn.benchmark = True

    # train
    print('start train......')
    epochs = 300
    best_acc = 0
    best_epoch = 0
    for epoch in range(0, epochs + 1):
        loss, train_acc = train(train_loader, model, criterion, optimizer)

        print('epoch:%d, loss: %.03f  | Acc: %.3f%%' % (epoch + 1, loss, train_acc))
        if epoch > 1:
            model.eval()
            print('waiting test')
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for idx, (inputs, label) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    label = torch.tensor(label, dtype=torch.long)
                    label = label.to(device)
                    # output = model(inputs)
                    output, _ = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    test_total += label.size(0)
                    test_correct += (predicted == label).sum()
            acc = 100. * test_correct / test_total
            print('the accuracy of test %.3f%%' % (acc))

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                if best_acc > 95:
                    torch.save(model.module.state_dict(), './save_model/IDNet/IDNet_acc={}.ckpt'.format(best_acc))
    print('best_epoch {},best_acc {}'.format(best_epoch, best_acc))
    print('finished train')


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_total = 0
    train_correct = 0
    loss_all = 0.0


    for idx, (inputs, label) in enumerate(train_loader):
        inputs = inputs.to(device)
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(device)
        # ===================forward=====================
        output, _ = model(inputs)
        loss = criterion(output, label).float()
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


if __name__ == '__main__':
    main()

