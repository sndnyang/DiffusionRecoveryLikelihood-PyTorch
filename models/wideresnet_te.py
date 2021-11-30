# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .wideresnet import get_norm, conv3x3, Identity
from .te_utils import get_timestep_embedding


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None, leak=.2):
        super(wide_basic, self).__init__()
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.temb_dense = nn.Linear(512, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        x, temb = x
        out = self.bn1(x)
        out = self.conv1(self.lrelu(out))
        if temb is not None:
            # add in timestep embedding
            temp_o = self.lrelu(self.temb_dense(temb))
            b, l = temp_o.shape
            out += temp_o.view(b, l, 1, 1)

        out = self.dropout(out)
        out = self.bn2(out)
        out = self.conv2(self.lrelu(out))
        out += self.shortcut(x)

        return out, temb


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, input_channels=3, sum_pool=False, norm=None, leak=.2, dropout_rate=0.0):
        super(Wide_ResNet, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)
        self.n_classes = num_classes

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d, time embedding' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.layer_one_out = None
        self.conv1 = conv3x3(input_channels, nStages[0])
        # self.layer_one = self.conv1
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, leak=leak)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, leak=leak)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, leak=leak)
        self.bn1 = get_norm(nStages[3], self.norm)
        self.last_dim = nStages[3]
        self.linear = nn.Linear(nStages[3], num_classes)
        self.temb_dense_0 = nn.Linear(128, 512)
        self.temb_dense_1 = nn.Linear(512, 512)
        self.temb_dense_2 = nn.Linear(512, nStages[3])

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, leak=0.2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, leak=leak, norm=self.norm))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, t, logits=False, feature=True):
        out = self.conv1(x)
        assert x.dtype == torch.float32
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(x.shape[0], dtype=torch.int64, device=x.device) * t
        temb = get_timestep_embedding(t, 128)
        temb = self.temb_dense_0(temb)
        temb = self.temb_dense_1(self.lrelu(temb))

        out, _ = self.layer1([out, temb])
        out, _ = self.layer2([out, temb])
        out, _ = self.layer3([out, temb])
        out = self.lrelu(self.bn1(out))
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            if self.n_classes > 100:
                out = F.adaptive_avg_pool2d(out, 1)
            else:
                out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        temb = self.temb_dense_2(self.lrelu(temb))
        out *= temb
        if logits:
            out = self.linear(out)
        return out


if __name__ == '__main__':
    ch_mult = (1, 2, 2, 2)
    # net = net_res_temb2(name='net', ch=128, ch_mult=ch_mult, num_res_blocks=2, attn_resolutions=(16,))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Wide_ResNet(28, 10, norm='batch', dropout_rate=0).to(device)
    x = torch.randn([64, 3, 32, 32]).to(device)
    # out = net(x, 0, 0)
    # print(out.shape)

    t = torch.randint(size=[64], high=6).to(device)
    output = net(x, temb=t)
    print(output.shape)
