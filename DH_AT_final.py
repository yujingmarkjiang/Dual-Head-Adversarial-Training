import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WRN_Module(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, sec_head=False):
        super(WRN_Module, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.sec_head = sec_head
        if not sec_head:
            # 1st conv before any network block
            self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                                   padding=1, bias=False)
            # 1st block
            self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        ############# The last layer of ReLU ############# 
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = x
        if not self.sec_head:
            out = self.conv1(out)
            out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def forward_block1(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        return out


class Small_CNN(nn.Module):
    def __init__(self, dropRate=0.0):
        super(Small_CNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(2, 16, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm1d(16)
        self.avg_pool = nn.AvgPool1d(16, stride=2)
        self.fc = nn.Linear(360*8, 10)

    def class_wise_permute(self, in_tensor):
        WIDTH = 8
        in_tensor = torch.transpose(in_tensor, 1, 2)
        shape = in_tensor.shape
        out = torch.zeros(shape[0], 2, 45 * WIDTH)
        index = 0
        for i in range(9):
            for j in range(i+1, 10):
                t1 = in_tensor[:, i, :].reshape(shape[0], 1, WIDTH)
                t2 = in_tensor[:, j, :].reshape(shape[0], 1, WIDTH)
                out[:, :, index:index+WIDTH] = torch.cat((t1, t2), 1)
                index += WIDTH
        return out.cuda()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        out = self.conv2(self.class_wise_permute(out))
        out = self.bn2(out)
        out = torch.transpose(out, 1, 2)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class WRN_Attach(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WRN_Attach, self).__init__()
        self.num_classes = num_classes
        self.WRN1 = WRN_Module(depth, num_classes, widen_factor, dropRate)
        self.WRN2 = WRN_Module(depth, num_classes, widen_factor, dropRate, sec_head=True)
        self.cnn = Small_CNN()

    def forward(self, x):
        out1 = self.WRN1(x).reshape([-1, 1, self.num_classes])
        out2 = self.WRN2(self.WRN1.forward_block1(x)).reshape([-1, 1, self.num_classes])
        return self.cnn(torch.cat((out1, out2), 1))
