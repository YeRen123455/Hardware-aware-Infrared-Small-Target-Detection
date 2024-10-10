

from search.utils.my_modules import *
from search.modules.layers import *

import json
import torch.nn.functional as F
import numpy as np

def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
    assert net_config is not None, 'Please input a network config'
    # net_config_path = download_url(net_config)
    net_config_path = '/media/gfkd/sda/NAS/proxylessnas-master/search/config/proxyless_mobile.config'
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut             = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res    = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])

        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)

    def get_latency(self, x):
        latency, conv_x = self.mobile_inverted_conv.get_latency(x)


        return latency, self.forward(x)

class ProxylessNASNets(MyNetwork):

    def __init__(self, blocks, post_transform_conv_block, model_name, iterations, add_decoder, add_encoder0, more_down):
        super(ProxylessNASNets, self).__init__()

        self.encoders           = nn.ModuleList(blocks)
        self.post_transform_conv_block = post_transform_conv_block
        # self.up                 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.pool               = nn.MaxPool2d(2, 2)
        self.model_name         = model_name
        self.iterations         = iterations
        self.add_decoder        = add_decoder
        self.add_encoder0       = add_encoder0
        self.more_down          = more_down


    #### res_add_new  给网络自适应地串接skip connection
    # def forward(self, x):
    #
    #     e_i        = 0
    #     this_layer = self.iterations
    #     for iteration in range(self.iterations):
    #         # print('this_layer：', this_layer)
    #         enc    = [None for i in range(this_layer)]
    #         if iteration == 0:
    #             encoder_0 = [None for i in range(this_layer)]
    #         # encoding path
    #         for layer in range(this_layer):
    #             if (layer == 0 and iteration == 0):
    #                 x_in   = x
    #             x_in       = self.encoders[e_i](x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (iteration != 0 and self.encoders[e_i].skipped == False) else [])
    #             encoder_0[layer] = x_in if iteration == 0 else encoder_0[layer]
    #             encoder_0[layer] = [] if ((iteration != 0) and self.encoders[e_i].skipped == False) else encoder_0[layer]
    #             enc[layer]       = x_in
    #             x_in             = F.max_pool2d(x_in, 2)
    #             e_i              = e_i + 1
    #         this_layer          -= 1
    #         enc_after            = enc
    #         x_in                 = enc_after[0]
    #     x = self.post_transform_conv_block(x_in)
    #     return x

    #### visualization 专用
    def forward(self, x):
        e_i        = 0
        this_layer = self.iterations
        vis_output = []
        for iteration in range(self.iterations):
            # print('this_layer：', this_layer)
            enc    = [None for i in range(this_layer)]
            if iteration == 0:
                encoder_0 = [None for i in range(this_layer)]
            # encoding path
            for layer in range(this_layer):
                if (layer == 0 and iteration == 0):
                    x_in   = x
                x_in       = self.encoders[e_i](x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (iteration != 0 and self.encoders[e_i].skipped == False) else [])
                encoder_0[layer] = x_in if iteration == 0 else encoder_0[layer]
                encoder_0[layer] = [] if ((iteration != 0) and self.encoders[e_i].skipped == False) else encoder_0[layer]
                enc[layer]       = x_in
                x_in             = F.max_pool2d(x_in, 2)
                e_i              = e_i + 1
            this_layer          -= 1
            enc_after            = enc
            x_in                 = enc_after[0]
            vis_output.append(enc_after[-1])
        x = self.post_transform_conv_block(x_in)
        return x, vis_output

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    def config(self):
        return {
            'name':              ProxylessNASNets.__name__,
            'bn':                self.get_bn_param(),
            'blocks':            [block.operator.config for block in self.encoders],
            'post_transform_conv_block': self.post_transform_conv_block.config,
            'model_name':        'Super_half_retrain'  if self.model_name == "Super_half_search" else self.model_name ,
            'add_decoder':       self.add_decoder,
            'add_encoder0':      self.add_encoder0,
            'iterations':        self.iterations,
            'more_down':         self.more_down,
        }

    def save_config(self):
        return {
            'name':              ProxylessNASNets.__name__,
            'bn':                self.get_bn_param(),
            'blocks':            [block.operator.config for block in self.encoders],
            'post_transform_conv_block': self.post_transform_conv_block.config,
            'model_name':        'Super_half_retrain'  if self.model_name == "Super_half_search" else self.model_name ,
            'add_decoder':       self.add_decoder,
            'add_encoder0':      self.add_encoder0,
            'iterations':        self.iterations,}



    @staticmethod
    def build_from_config(config, gene_path):
        from search.models.super_nets.super_proxyless_SIRST import Block, BFS, decode

        def reset_gene(encoders, genes, iterations, skip_genes=None):
            count = 0
            for idx, gene in enumerate(genes):
                for level in range(iterations - idx):
                    try:
                        encoders[count].skip_pre = gene[level]
                    except:
                        print('reset_gene_error')
                    encoders[count].skipped  = (skip_genes is not None) and skip_genes[idx][level]
                    if encoders[count].skipped:
                        print('find one encoder skipped! at iteration ' + str(idx) + ' level ' + str(level))
                    count += 1
            return encoders

        post_transform_conv_block = set_layer_from_config(config['post_transform_conv_block'])

        model_name       = config['model_name']
        add_decoder      = config['add_decoder']
        add_encoder0     = config['add_encoder0']
        iterations       = config['iterations']
        more_down        = config['more_down']


        blocks           = nn.ModuleList()
        conv_blocks      = []
        conv_blocks_all  = []
        count_conv       = 0

        if model_name == 'Super_all':
            for i in range(len(config['blocks'])):
                if config['blocks'][i]['name']   == 'ResConvLayer':
                    conv_blocks_all = ResConvLayer.build_from_config(config['blocks'][i])
                    block = Block(config['post_transform_conv_block']['in_channels'], config['post_transform_conv_block']['in_channels'], conv_blocks_all=conv_blocks_all)
                    blocks.append(block)

                elif config['blocks'][i]['name'] == 'SpaConvLayer':
                    conv_blocks_all = SpaConvLayer.build_from_config(config['blocks'][i])
                    block = Block(config['post_transform_conv_block']['in_channels'],
                                  config['post_transform_conv_block']['in_channels'], conv_blocks_all=conv_blocks_all)
                    blocks.append(block)

                elif config['blocks'][i]['name'] == 'MBInvertedConvLayer':
                    conv_blocks_all = MBInvertedConvLayer.build_from_config(config['blocks'][i])
                    block = Block(config['post_transform_conv_block']['in_channels'], config['post_transform_conv_block']['in_channels'], conv_blocks_all=conv_blocks_all)
                    blocks.append(block)


        elif 'Super_half' in model_name:
            for iteration in range(iterations):
                for layer in range(iterations - iteration):
                    if iteration == 0:
                        if   config['blocks'][layer]['name']   == 'ResConvLayer':
                            conv_blocks.append(ResConvLayer.build_from_config(config['blocks'][layer]))
                        elif   config['blocks'][layer]['name'] == 'SpaConvLayer':
                            conv_blocks.append(SpaConvLayer.build_from_config(config['blocks'][layer]))
                        elif config['blocks'][layer]['name']   == 'MBInvertedConvLayer':
                            conv_blocks.append(MBInvertedConvLayer.build_from_config(config['blocks'][layer]))
                    if   config['blocks'][layer]['name'] == 'ResConvLayer':
                         conv_blocks_all.append(ResConvLayer.build_from_config(config['blocks'][layer]))
                    elif config['blocks'][layer]['name'] == 'SpaConvLayer':
                        conv_blocks_all.append(SpaConvLayer.build_from_config(config['blocks'][layer]))
                    elif config['blocks'][layer]['name'] == 'MBInvertedConvLayer':
                         conv_blocks_all.append(MBInvertedConvLayer.build_from_config(config['blocks'][layer]))


                    block = Block(config['post_transform_conv_block']['in_channels'], config['post_transform_conv_block']['in_channels'],
                                  conv_blocks[layer], iteration=iteration, mode=model_name, conv_blocks_all=conv_blocks_all[count_conv])
                    count_conv += 1
                    blocks.append(block)


        genes             = decode(gene_path, iterations)
        skip_codes        = BFS(iterations, genes, add_decoder, add_encoder0)  # check if a block is skipped or not
        blocks            = reset_gene(blocks, genes, iterations, skip_codes)


        net = ProxylessNASNets(blocks, post_transform_conv_block, model_name, iterations, add_decoder, add_encoder0, more_down)
        # if 'bn' in config:
        #     net.set_bn_param(**config['bn'])
        # else:
        #     net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):

        flop = 0
        e_i  = 0
        total_params = 0
        this_layer = self.iterations
        for iteration in range(self.iterations):
            # print('this_layer：', this_layer)
            enc = [None for i in range(this_layer)]
            if iteration ==0:
                encoder_0 = [None for i in range(this_layer)]
            # encoding path
            for layer in range(this_layer):
                if (layer == 0 and iteration == 0):
                    x_in = x
                # delta_flop, parames, x_in  = self.encoders[e_i].Block_flops(x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (layer == (this_layer-1) and iteration!=0) else [])

                delta_flop, parames, x_in = self.encoders[e_i].Block_flops(x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (iteration != 0 and self.encoders[e_i].skipped == False) else [])
                encoder_0[layer] = x_in if iteration == 0 else encoder_0[layer]
                encoder_0[layer] = [] if ((iteration != 0) and self.encoders[e_i].skipped == False) else encoder_0[layer]

                flop          += delta_flop
                total_params  += parames
                # print('delta_flop: ', delta_flop)
                if iteration == 0:
                    encoder_0[layer] = x_in
                enc[layer]        = x_in
                x_in              = F.max_pool2d(x_in, 2)
                e_i               = e_i + 1
            this_layer -= 1
            enc_after   = enc
            x_in        = enc_after[0]

        delta_flop, parames, x = self.post_transform_conv_block.get_flops(x_in)
        # print('delta_flop: ', delta_flop)

        flop         += delta_flop
        total_params += parames
        print('flop: ', flop/10**9, 'GFLOPs')

        return flop, total_params, x

