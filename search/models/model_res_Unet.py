import torch
import torch.nn as nn
from   thop import profile, clever_format

def conv1x1(in_channels, out_channels, stride=1):
    ''' 1x1 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    ''' 3x3 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)

def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    ''' 7x7 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, dilation=dilation, bias=False)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class BAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio, dilation):
        super(BAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=in_channel, out_features=self.hid_channel)
        self.bn1_1d = nn.BatchNorm1d(self.hid_channel)
        self.fc2 = nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        self.bn2_1d = nn.BatchNorm1d(in_channel)

        self.conv1 = conv1x1(in_channel, self.hid_channel)
        self.bn1_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv2 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn2_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv3 = conv3x3(self.hid_channel, self.hid_channel, stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn3_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv4 = conv1x1(self.hid_channel, 1)
        self.bn4_2d = nn.BatchNorm2d(1)

    def forward(self, x):
        # Channel attention
        Mc = self.globalAvgPool(x)
        Mc = Mc.view(Mc.size(0), -1)

        Mc = self.fc1(Mc)
        Mc = self.bn1_1d(Mc)
        Mc = self.relu(Mc)

        Mc = self.fc2(Mc)
        Mc = self.bn2_1d(Mc)
        Mc = self.relu(Mc)

        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)

        # Spatial attention
        Ms = self.conv1(x)
        Ms = self.bn1_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv2(Ms)
        Ms = self.bn2_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv3(Ms)
        Ms = self.bn3_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv4(Ms)
        Ms = self.bn4_2d(Ms)
        Ms = self.relu(Ms)

        Ms = Ms.view(x.size(0), 1, x.size(2), x.size(3))
        Mf = 1 + self.sigmoid(Mc * Ms)
        return x * Mf

class res_UNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, layer=4):
        super(res_UNet, self).__init__()
        self.ratio    = 4
        self.dilation = 4
        self.pool     = nn.MaxPool2d(2, 2)
        self.up       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer    = layer
        if block    =='Res_block':
            block = Res_block
            self.block = 'Res_block'

        elif block  =='Res_CBAM_block':
            block = Res_CBAM_block
            self.block = 'Res_CBAM_block'

        elif block  =='BAM':
            block = Res_block
            self.block = 'BAM'

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])


        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_2 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_3 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])


        self.bam1 = BAM(nb_filter[0], self.ratio, self.dilation)
        self.bam2 = BAM(nb_filter[1], self.ratio, self.dilation)
        self.bam3 = BAM(nb_filter[2], self.ratio, self.dilation)
        self.bam4 = BAM(nb_filter[3], self.ratio, self.dilation)

        self.final   = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.layer ==4:
            x0_0 = self.conv0_0(input)
            if self.block == 'BAM':
                x0_0 = self.bam1(x0_0)
            x1_0 = self.conv1_0(self.pool(x0_0))
            if self.block == 'BAM':
                x1_0 = self.bam2(x1_0)
            x2_0 = self.conv2_0(self.pool(x1_0))
            if self.block == 'BAM':
                x2_0 = self.bam3(x2_0)
            x3_0 = self.conv3_0(self.pool(x2_0))
            if self.block == 'BAM':
                x3_0 = self.bam4(x3_0)
            x4_0 = self.conv4_0(self.pool(x3_0))

            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
            output = self.final(x0_4)
            return output

        elif self.layer ==3:
            x0_0 = self.conv0_0(input)
            if self.block == 'BAM':
                x0_0 = self.bam1(x0_0)
            x1_0 = self.conv1_0(self.pool(x0_0))
            if self.block == 'BAM':
                x1_0 = self.bam2(x1_0)
            x2_0 = self.conv2_0(self.pool(x1_0))
            if self.block == 'BAM':
                x2_0 = self.bam3(x2_0)
            x3_0 = self.conv3_0(self.pool(x2_0))

            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))

            output = self.final(x0_3)
            return output

#####################################
### FLops, Params, Inference time evaluation
if __name__ == '__main__':
    from model.load_param_data import  load_param
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    nb_filter, num_blocks, ACM_channel, ACM_layer_blocks = load_param('two', 'resnet_18')
    input = torch.randn(2, 3, 256, 256).cuda()
    in_channels=3
    # model = res_UNet(num_classes=1, input_channels=in_channels, block=Res_block, num_blocks=num_blocks, nb_filter=nb_filter)
    model = res_UNet( num_classes=1, input_channels=in_channels, block='Res_block', num_blocks=num_blocks, nb_filter=nb_filter, layer=3)

    model = model.cuda()
    flops, params = profile(model, inputs=(input,), verbose=True)
    # flops, params = clever_format([flops, params], "%.3f")
    start_time = time.time()
    output     = model(input)
    end_time   = time.time()
    print('flops:', flops/(10**9), 'params:', params/(10**6))
    print('inference time per image:',end_time-start_time )


