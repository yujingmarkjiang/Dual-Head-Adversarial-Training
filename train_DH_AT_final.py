from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

from models.DH_AT_final import *
from trades import trades_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', type=float, default=3.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-Merged',
                    help='directory of model for saving checkpoint')
parser.add_argument('--model_path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for @white-box@ attack evaluation')
parser.add_argument('--model_path2',
                    default='./checkpoints/model-wideres_34_10_Beta-3.0-epoch076.pt',
                    help='model for @white-box@ attack evaluation')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--wrn_depth', default=34, type=int,
                    help='depth of WideResNet')
parser.add_argument('--num_class', default=10, type=int,
                    help='the number of classes used in training and testing')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

EXCLUDE_LIST = [0,1,2,3,4,5,6,7,8,9][args.num_class:]

# Process training set, excluding certain classes
labels = np.array(trainset.targets)
exclude = np.array(EXCLUDE_LIST).reshape(1, -1)
mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

trainset.data = trainset.data[mask]
trainset.targets = labels[mask].tolist()

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

print(f"Class label    min: {min(trainset.targets)}, max: {max(trainset.targets)}")

# Process test set, excluding certain classes
labels = np.array(testset.targets)
exclude = np.array(EXCLUDE_LIST).reshape(1, -1)
mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

testset.data = testset.data[mask]
testset.targets = labels[mask].tolist()

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    model.module.WRN1.requires_grad = False
    model.module.WRN2.requires_grad = False
    model.module.cnn.requires_grad = True
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, adv_lbl = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)	
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), flush=True)


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)), flush=True)
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 25:
        lr = args.lr * 0.1
    if epoch >= 40:
        lr = args.lr * 0.01
    if epoch >= 50:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_merged_model():
    model = WRN_Attach(depth=args.wrn_depth, num_classes=args.num_class)
    sd1 = torch.load(args.model_path)
    sd2 = torch.load(args.model_path2)

    sd_p = {'WRN1.'+k.strip('module.'):v for k,v in sd1.items()}
    sd_p.update({'WRN2.'+k.strip('module.'):v for k,v in sd2.items()})

    model.load_state_dict(sd_p, strict=False)

    print(args.model_path, flush=True)
    print(args.model_path2, flush=True)
    
    return model, sd_p


def main2():

    model, sd_p = load_merged_model()
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        #args.batch_size = args.batch_size * torch.cuda.device_count()
        print("batch_size = %d" % args.batch_size, flush=True)
    
    model.to(device)

    start_epoch = 1

    print(args.model_dir, flush=True)

    optimizer = optim.SGD(model.module.cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
                torch.save(model.state_dict(),
                                   os.path.join(model_dir, 'model-merged_%d_%d_Beta-%.1f-epoch%03d.pt' % (args.wrn_depth, 
                                           args.num_class, args.beta, epoch)))
                torch.save(optimizer.state_dict(),
                                   os.path.join(model_dir, 'opt-merged_%d_%d_Beta-%.1f-checkpoint_epoch%03d.tar' % (args.wrn_depth, 
                                           args.num_class, args.beta, epoch)))

def main():
    model = WRN_Attach()
    
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        #args.batch_size = args.batch_size * torch.cuda.device_count()
        print("batch_size = %d" % args.batch_size, flush=True)
    
    model.to(device)
    
    return model


def params(model):
    params = sum(p.numel() for p in model.parameters())
    return params

if __name__ == "__main__":
    model = main()
    #for batch_idx, (x, y) in enumerate(train_loader):
    #    break
    from models.wideresnet import WideResNet
    m1 = WideResNet(28,10,10)
    m2 = WideResNet(34,10,10)
    m3 = WideResNet(34,10,20)
    m4 = WideResNet(70,10,16)
    m5 = WideResNet(106,10,8)
    #torch.multiprocessing.freeze_support()
    
