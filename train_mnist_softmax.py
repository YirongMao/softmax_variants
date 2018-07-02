import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from model_utils import CenterLoss
from model_utils import Net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 100
num_epoch = 100


def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    #   plt.xlim(xmin=-5,xmax=5)
    #   plt.ylim(ymin=-5,ymax=5)
    plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    plt.savefig('./images/softmax_loss_epoch=%d.jpg' % epoch)
    plt.close()


def test(test_loder, model, use_cuda):
    correct = 0
    total = 0
    for i, (data, target) in enumerate(test_loder):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        ip1, logits = model(data)
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target.data).sum()

    print('Test Accuracy of the model on the 10000 test images: %f %%' % (100 * correct / total))


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    ip1_loader = []
    idx_loader = []
    for i, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        feats, logits = model(data)
        loss = criterion[0](logits, target)

        _, predicted = torch.max(logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()

        ip1_loader.append(feats)
        idx_loader.append((target))
        if (i + 1) % 50 == 0:
            print('Epoch [%d], Iter [%d/%d] Loss: %.4f Acc %.4f'
                  % (epoch, i + 1, len(train_loader) // batch_size, loss.data[0], accuracy))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)


def main():
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    # Dataset
    trainset = datasets.MNIST('./data/', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

    testset = datasets.MNIST('./data/', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = DataLoader(testset, batch_size=100, shuffle=True, num_workers=4)

    # Model
    model = Net()

    # NLLLoss
    nllloss = nn.CrossEntropyLoss()
    if use_cuda:
        nllloss = nllloss.cuda()
        model = model.cuda()
    criterion = [nllloss]

    # optimzer4nn
    optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    sheduler = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.8)

    for epoch in range(num_epoch):
        sheduler.step()
        # print optimizer4nn.param_groups[0]['lr']
        train(train_loader, model, criterion, [optimizer4nn], epoch + 1, use_cuda)
        torch.save(model, './model/softmax_net_' + str(epoch + 1) + '.model')
        test(test_loader, model, use_cuda)


if __name__ == '__main__':
    main()