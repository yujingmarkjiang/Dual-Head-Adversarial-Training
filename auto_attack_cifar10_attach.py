import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from models.wideresnet_attach import *
from attacks_bl.autoattack import AutoAttack

import sys
sys.path.insert(0,'..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=0.031)
    parser.add_argument('--model_path', type=str, default='./model-cifar-wrn_baseline/model_cifar_wrn.pt')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    
    
    args = parser.parse_args()

    print('AutoAttack')

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        print("batch_size = %d" % args.batch_size)

    # load model
    model = WRN_Attach(depth=34, num_classes=10)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    if not args.multi_gpu:
        model = nn.DataParallel(model)
    model.eval()

    print(args.model_path, flush=True)

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)
    
    # load attack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=None,
        version=args.version)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    # run attack and save images
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
            bs=args.batch_size)
            
                
