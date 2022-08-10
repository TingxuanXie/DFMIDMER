from models.SingleResNet18_exp import *  # change
from models.SingleResNet18_id import *
from torchvision import transforms
import transforms.temporal_transforms as TT
from dataset.load_data import load_data  # change
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
from losses import DivRegLoss
from models.KLloss import kl_loss
from dataset.load_data_id import load_data_id


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IDNet = SingleResNet18_id(num_classes=73).to(device)
IDNet.load_state_dict(torch.load('IDNet.ckpt'))
IDNet = torch.nn.DataParallel(IDNet, device_ids=[0,1,2,3])



def main():

    classes = 6
    train_file = r'/data/txt_file/train_label_0328.txt'
    test_file = r'/data/txt_file/SDU_test_label_0328.txt'
    root = r'/data/SDU/'

    spatial_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    temporal_transform = TT.TemporalRandomCrop(size=6, stride=4)

    train_set = load_data(file=train_file, root=root,  # change
                          transform=spatial_transform,  # change
                          temporal_transform=temporal_transform)
    test_set = load_data(file=test_file, root=root,  # change
                             transform=spatial_transform,  # change
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
    model = SingleResNet18_exp(num_classes=classes)
    model = model.to(device)
    print('model\n')
    print(model)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    # set the criterion
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_div = DivRegLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    cudnn.benchmark = True

    # train
    print('start train......')
    epochs = 300
    best_acc = 0
    best_epoch = 0
    for epoch in range(0, epochs + 1):
        loss, train_acc = train(train_loader, model, criterion, criterion_div, optimizer)

        print('epoch:%d, loss: %.03f  | Acc: %.3f%%' % (epoch + 1, loss, train_acc))
        if epoch > 1:
            model.eval()
            print('waiting test')
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for idx, (inputs, label, index) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    label = torch.tensor(label, dtype=torch.long)
                    label = label.to(device)
                    # output = model(inputs)
                    output, _, _ = model(inputs)
                    _, predicted = torch.max(output.data, 1)
                    test_total += label.size(0)
                    test_correct += (predicted == label).sum()
            acc = 100. * test_correct / test_total
            print('the accuracy of test %.3f%%' % (acc))

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
    print('best_epoch {},best_acc {}'.format(best_epoch, best_acc))
    print('finished train')


def train(train_loader, model, criterion, criterion_div, optimizer):
    model.train()
    train_total = 0
    train_correct = 0
    loss_all = 0.0
    loss_cross_all = 0.0
    loss_div_all = 0.0
    loss_kl_all = 0.0

    for idx, (inputs, label, index) in enumerate(train_loader):
        inputs = inputs.to(device)
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(device)
        # ===================forward=====================
        output, masks, x_exp = model(inputs)
        # _, x_id = IDNet(inputs)
        # print('x_exp', x_exp.shape, 'x_id', x_id.shape)
        loss_cross = criterion(output, label).float()
        loss_div = criterion_div(masks).float()
        # loss_kl = kl_loss(x_exp, x_id).float()
        loss = loss_cross + 0.8*loss_div
        # print('loss=', loss, 'loss_cross', loss_cross, 'loss_div', loss_div)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================meters=====================
        loss_cross_all += loss_cross
        loss_div_all += loss_div
        # loss_kl_all += loss_kl
        loss_all += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum()
    train_acc = 100. * train_correct / train_total
    print('loss=', loss_all, 'loss_cross', loss_cross_all, 'loss_div', loss_div_all, 'loss_kl', loss_kl_all)

    return loss_all, train_acc


if __name__ == '__main__':
    main()

