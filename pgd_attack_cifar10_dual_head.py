from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet_dual_head import *
from models.wideresnet import WideResNet as WRN_BL
from models.resnet import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model_path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for @white-box@ attack evaluation')
parser.add_argument('--source_model_path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for #black-box# attack evaluation')
parser.add_argument('--target_model_path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for #black-box# attack evaluation')
parser.add_argument('--white_box_attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--black_box_attack', default=False,
                    help='whether perform black-box attack')
parser.add_argument('--wrn_depth', default=34, type=int,
                    help='depth of WideResNet')
parser.add_argument('--num_class', default=10, type=int,
                    help='the number of classes used in training and testing')
parser.add_argument('--model_bl_path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='baseline model for @white-box@ attack evaluation')
parser.add_argument('--multi_gpu', default=True,
                    help='Multiple GPUs are used in training')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


EXCLUDE_LIST = [0,1,2,3,4,5,6,7,8,9][args.num_class:]

transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

# Process test set, excluding certain classes
labels = np.array(testset.targets)
exclude = np.array(EXCLUDE_LIST).reshape(1, -1)
mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

testset.data = testset.data[mask]
testset.targets = labels[mask].tolist()

print(f"Class label    min: {min(testset.targets)}, max: {max(testset.targets)}")

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)[0]
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd)[0], y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd)[0].data.max(1)[1] != y.data).float().sum()
    print('err, err_pgd (white-box): ', err, err_pgd, flush=True)
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    data_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
        data_total += len(target)
    print('natural_err_total: ', natural_err_total, f"({natural_err_total/data_total})")
    print('robust_err_total: ', robust_err_total, f"({robust_err_total/data_total})")
    print("data_total:", data_total)


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()

    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        #model = WideResNet().to(device)
        model = WideResNet(depth=args.wrn_depth, num_classes=args.num_class)
        if args.multi_gpu:
            model = nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(torch.load(args.model_path))

        # load the model to generate PDG adv. data
        #model_bl = WRN_BL(depth=args.wrn_depth, num_classes=args.num_class).to(device)
        #model_bl.load_state_dict(torch.load(args.model_bl_path))

        #eval_adv_test_whitebox(model, model_bl, device, test_loader)
        eval_adv_test_whitebox(model, device, test_loader)
    if args.black_box_attack:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
