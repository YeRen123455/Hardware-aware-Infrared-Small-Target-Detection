

from search.utils import *
from collections import OrderedDict
import time
from torch.nn import GroupNorm, Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer=None, group=1, stride = 1):
        super(Res_block, self).__init__()
        self.conv1        = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = int(kernel_size/2), groups=group)
        self.out_channels = out_channels

        self.layer = layer
        if   layer == 'first':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2), groups=group)
            self.bn1   = nn.BatchNorm2d(out_channels)
            self.relu1  = nn.ReLU(inplace = True)
        elif layer == 'final':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2), groups=group)
            self.out_channels = out_channels
        elif layer == 'middle':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2), groups=group)
            self.bn1   = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU(inplace = True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = int(kernel_size/2), groups=group)
            self.bn2   = nn.BatchNorm2d(out_channels)
            if stride != 1 or out_channels != in_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=group),
                    nn.BatchNorm2d(out_channels))
            else:
                self.shortcut = None
            self.relu2  = nn.ReLU(inplace = True)

    def forward(self, x):
        if   self.layer == 'first':
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
        elif self.layer == 'final':
            out = self.conv1(x)
        elif self.layer == 'middle':
            residual = x
            if self.shortcut is not None:
                residual = self.shortcut(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += residual
            out = self.relu2(out)

        try:
            return out
        except:
            print('Res_load_error')

class Spa_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, group=1, layer=None, stride=1):
        super(Spa_block, self).__init__()
        self.layer = layer

        if   layer == 'first':
            self.convblock_1  = Sequential(nn.Conv2d(in_channels,  out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(int(kernel_size / 2), 0), groups=group),
                                           nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, int(kernel_size / 2)), groups=group))
            self.bn1          = nn.BatchNorm2d(out_channels)
            self.relu1        = nn.ReLU(inplace = True)
        elif layer == 'final':
            self.convblock_1  = Sequential(nn.Conv2d(in_channels, out_channels,  kernel_size=(kernel_size, 1), stride=stride, padding=(int(kernel_size / 2), 0), groups=group),
                                           nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, int(kernel_size / 2)), groups=group))
            self.out_channels = out_channels
        elif layer == 'middle':
            self.convblock_1  = Sequential(nn.Conv2d(in_channels, out_channels,  kernel_size=(kernel_size, 1), stride=stride, padding=(int(kernel_size / 2), 0), groups=group),
                                           nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, int(kernel_size / 2)), groups=group))
            self.bn1          = nn.BatchNorm2d(out_channels)
            self.relu1        = nn.ReLU(inplace = True)
            self.convblock_2  = Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(int(kernel_size / 2), 0), groups=group),
                                           nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), stride=stride, padding=(0, int(kernel_size / 2)), groups=group))
            self.bn2          = nn.BatchNorm2d(out_channels)
            if stride != 1 or out_channels != in_channels:
                self.shortcut = nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=group), nn.BatchNorm2d(out_channels))
            else:
                self.shortcut = None
            self.relu2  = nn.ReLU(inplace = True)

    def forward(self, x):
        if   self.layer == 'first':
            out = self.convblock_1(x)
            out = self.bn1(out)
            out = self.relu(out)
        elif self.layer == 'final':
            out = self.convblock_1(x)
        elif self.layer == 'middle':
            residual = x
            if self.shortcut is not None:
                residual = self.shortcut(x)
            out = self.convblock_1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.convblock_2(out)
            out = self.bn2(out)
            out += residual
            out = self.relu2(out)
        try:
            return out
        except:
            print('Res_load_error')


class MBInverted_block(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dim, kernel_size, expand_ratio, stride=1, layer=None):
        super(MBInverted_block, self).__init__()
        if expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn',   nn.BatchNorm2d(feature_dim)),
                ('act',  nn.ReLU6(inplace=True))]))

        self.pad = get_same_padding(kernel_size)
        self.depth_conv   = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, self.pad, groups=feature_dim, bias=False)),
            ('bn',   nn.BatchNorm2d(feature_dim)),
            ('act',  nn.ReLU6(inplace=True))]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels)),
            ('act', nn.ReLU6(inplace=True))]))


    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__:           ConvLayer,
        DepthConvLayer.__name__:      DepthConvLayer,
        PoolingLayer.__name__:        PoolingLayer,
        IdentityLayer.__name__:       IdentityLayer,
        LinearLayer.__name__:         LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ZeroLayer.__name__:           ZeroLayer,
        ResConvLayer.__name__:        ResConvLayer,

    }

    # layer_name = layer_config.pop('name')
    # layer      = name2layer[layer_name]
    # return layer.build_from_config(layer_config)

    layer_name = layer_config.pop('name')
    layer      = name2layer[layer_name]
    return layer.build_from_config(layer_config)

class My2DLayer(MyModule):

    def __init__(self, in_channels, out_channels, kernel_size,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act',  layer=None):
        super(My2DLayer, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.use_bn       = use_bn
        self.act_func     = act_func
        self.dropout_rate = dropout_rate
        self.ops_order    = ops_order
        self.layer        = layer

        self.conv_block = self._make_layer(Res_block, in_channels, out_channels, kernel_size, num_blocks=1, layer =self.layer )


    def _make_layer(self, block, input_channels, output_channels, kernel_size, num_blocks=1, layer =None ):
        layers = []
        layers.append(block(input_channels, output_channels, kernel_size, layer))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels, kernel_size, layer))
        return nn.Sequential(*layers)



    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        # for module in self._modules.values():
        #     x = module(x)
        x = self.conv_block(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels':  self.in_channels,
            'out_channels': self.out_channels,
            'layer': self.layer,

        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn  = True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', layer=None):
        self.kernel_size = kernel_size
        self.stride      = stride
        self.dilation    = dilation
        self.groups      = groups
        self.bias        = bias
        self.has_shuffle = has_shuffle
        self.layer       = layer

        super(ConvLayer, self).__init__(in_channels, out_channels, self.kernel_size, use_bn, act_func, dropout_rate, ops_order, self.layer)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding    *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        # weight_dict['conv'] = nn.Conv2d(
        #     self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
        #     dilation=self.dilation, groups=self.groups, bias=self.bias
        # )

        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding
        )

        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                return '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                return '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                return '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self, x)[0], count_conv_flop(self, x)[1] , self.forward(x)

    def get_flops_all(self, x):
        final_flops = []
        this_layer_flops,_ = count_conv_flop(self, x)
        final_flops.append(this_layer_flops)
        return final_flops, self.forward(x)


    def get_latency(self, x):

        latency_list  = []
        start_time    = time.time()
        result        = self.forward(x)
        end_time      = time.time()
        latency_list.append(end_time-start_time)

        return latency_list, self.forward(x)


class DepthConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        self.kernel_size = kernel_size
        self.stride      = stride
        self.dilation    = dilation
        self.groups      = groups
        self.bias        = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding    *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['depth_conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.in_channels, bias=False
        )
        weight_dict['point_conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            return '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            return '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        x          = self.depth_conv(x)
        point_flop = count_conv_flop(self.point_conv, x)
        x          = self.point_conv(x)
        return depth_flop + point_flop, self.forward(x)


class PoolingLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        self.pool_type   = pool_type
        self.kernel_size = kernel_size
        self.stride      = stride

        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class IdentityLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class LinearLayer(MyModule):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False

class SpaConvLayer(MyModule):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, num_blocks=0, group=1, layer=None):
        super(SpaConvLayer, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.conv_block   = self._make_layer(Spa_block, in_channels, out_channels, self.kernel_size, num_blocks=num_blocks, group=group, layer=layer)
        self.num_blocks   = num_blocks
        self.group        = group
        self.layer       = layer

    def _make_layer(self, block, input_channels, output_channels, kernel_size, num_blocks=1, group=1, layer=None):
        layers = []
        layers.append(block(input_channels, output_channels, kernel_size, group, layer))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels, kernel_size, group,layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x

    def module_str(self):
        return '%dx%d_SpaConv' % (self.kernel_size, self.kernel_size)

    @property
    def config(self):
        return {
            'name':         SpaConvLayer.__name__,
            'in_channels':  self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size':  self.kernel_size,
            'stride':       self.stride,
            'num_blocks':   self.num_blocks,
            'group':        self.group,
            'layer':        self.layer}

    @staticmethod
    def build_from_config(config):
        return SpaConvLayer(config['in_channels'], config['out_channels'], config['kernel_size'], config['stride'],config['num_blocks'],config['group'],config['layer'])

    def get_flops(self, x):

        out_h     = int(x.size()[2] / self.stride)
        out_w     = int(x.size()[3] / self.stride)
        delta_ops = 0
        pararms   = 0
        for name, param in self.named_parameters():
            if ('weight' in name and len(np.shape(param)) > 1):
                out_channels, in_channels, kernel_size_1, kernel_size_2 = np.shape(param)
                delta_ops += out_channels * in_channels * kernel_size_1 * kernel_size_2 * out_h * out_w
                # print(out_channels, in_channels, kernel_size_1, kernel_size_2)
                # print(out_channels * in_channels * kernel_size_1 * kernel_size_2 * out_h * out_w)
                pararms   += out_channels * in_channels * kernel_size_1 * kernel_size_2
        x     = self.conv_block(x)
        return delta_ops, pararms, x

    def get_latency(self, x):
        latency_list = []
        start_time    = time.time()
        result        = self(x)
        end_time      = time.time()
        latency_list.append(end_time-start_time)
        return latency_list, self.forward(x)


    @staticmethod
    def is_zero_layer():
        return False

class ResConvLayer(MyModule):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, num_blocks=0, group=1, layer=None):
        super(ResConvLayer, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.conv_block   = self._make_layer(Res_block, in_channels, out_channels, self.kernel_size, num_blocks=num_blocks, group=group , layer=layer)
        self.num_blocks   = num_blocks
        self.group        = group
        self.layer        = layer
        # print('self.layer: ',self.layer)

    def _make_layer(self, block, input_channels, output_channels, kernel_size, num_blocks=1, group=1, layer=None):
        layers = []
        layers.append(block(input_channels, output_channels, kernel_size, layer, group, ))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels, kernel_size,layer, group, ))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_block(x)

        return x


    def module_str(self):
        if   self.group ==1:
             output  = '%dx%d_ResConv' % (self.kernel_size, self.kernel_size)
        elif self.group !=1:
            output = '%dx%d_GroupConv%d' % (self.kernel_size, self.kernel_size, self.group)

        return output

    @property
    def config(self):
        return {
            'name':         ResConvLayer.__name__,
            'in_channels':  self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size':  self.kernel_size,
            'stride':       self.stride,
            'num_blocks':   self.num_blocks,
            'group':        self.group,
            'layer':        self.layer,
        }

    @staticmethod
    def build_from_config(config):
        try:
            return ResConvLayer(config['in_channels'], config['out_channels'], config['kernel_size'], config['stride'], config['num_blocks'], config['group'], config['layer'])
        except:
            print('ResConvLayer_build_from_config_error')

    def get_flops(self, x):

        out_h     = int(x.size()[2] / self.stride)
        out_w     = int(x.size()[3] / self.stride)
        delta_ops = 0
        parames   = 0
        for name, param in self.named_parameters():
            if ('weight' in name and len(np.shape(param)) > 1):
                out_channels, in_channels, kernel_size_1, kernel_size_2 = np.shape(param)
                delta_ops += out_channels * in_channels * kernel_size_1 * kernel_size_2 * out_h * out_w
                # print( out_channels, in_channels, kernel_size_1, kernel_size_2, out_h, out_w)
                # print(out_channels * in_channels * kernel_size_1 * kernel_size_2 * out_h * out_w)
                parames   += out_channels * in_channels * kernel_size_1 * kernel_size_2
        x     = self.conv_block(x)
        return delta_ops, parames, x

    def get_latency(self, x):
        latency_list = []
        start_time    = time.time()
        result        = self(x)
        end_time      = time.time()
        latency_list.append(end_time-start_time)
        return latency_list, self.forward(x)


    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6, num_blocks=1, mid_channels=None, ):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.expand_ratio = expand_ratio
        self.num_blocks   = num_blocks
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels


        self.conv_block = self._make_layer(MBInverted_block, in_channels, out_channels, feature_dim, self.kernel_size, self.expand_ratio, num_blocks=num_blocks)


    def _make_layer(self, block, input_channels, output_channels, feature_dim, kernel_size, expand_ratio, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels, feature_dim, kernel_size, expand_ratio))
        for i in range(num_blocks - 1):
            if  self.expand_ratio ==1:
                layers.append(block(output_channels, output_channels, output_channels,  kernel_size, expand_ratio))
            else:
                layers.append(block(output_channels, output_channels, feature_dim,      kernel_size, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        # if self.inverted_bottleneck:
        #     x = self.inverted_bottleneck(x)
        # x = self.depth_conv(x)
        # x = self.point_linear(x)
        x = self.conv_block(x)

        return x

    def module_str(self):
        if   self.expand_ratio ==1:
            output = '%dx%d_DepthSC' % (self.kernel_size, self.kernel_size)

        elif self.expand_ratio !=1:
            output = '%dx%d_MBConv%d' % (self.kernel_size, self.kernel_size, self.expand_ratio)


        return output

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels':  self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size':  self.kernel_size,
            'stride':       self.stride,
            'num_blocks':   self.num_blocks,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):

        return MBInvertedConvLayer(config['in_channels'],   config['out_channels'],   config['kernel_size'],   config['stride'],
                                   config['expand_ratio'],  config['num_blocks'] ,     config['mid_channels'])

    def get_flops(self, x):
        flop, params = count_conv_flop(self, x)
        x    = self(x)
        return flop, params, x

    def get_latency(self, x):
        latency_list = []
        start_time    = time.time()
        result        = self(x)
        end_time      = time.time()
        latency_list.append(end_time-start_time)
        return latency_list, self.forward(x)


    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(MyModule):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True
