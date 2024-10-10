
from   queue    import Queue
import copy

import torch
from   torch.nn import GroupNorm, Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid
from   search.modules.mix_op import *
from   search.models.normal_nets.proxyless_nets import *
from   search.utils.latency_estimator import LatencyEstimator
from   search.utils import make_divisible
from   search.utils.utils    import BFS, decode, iter_layer_compute, load_param


class Block(Module):
    """
    Phase2 block
    """
    def __init__(self, in_channel, out_channel, conv=None, conv_type=None, skip_pre=None, skipped=False, iteration=None, mode=None, conv_blocks_all=None):
        super(Block, self).__init__()
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.conv_type   = conv_type
        self.iteration   = iteration
        self.operator    = conv_blocks_all


        if skip_pre is not None:
            skip_pre = list(set(skip_pre))
        else:
            skip_pre = [1,2,3,4]
        self.skip_pre    = skip_pre
        self.skipped     = skipped
        self.interpolate = None
        # print('self.interpolate： ', self.interpolate)

        self.mode        = mode

        if self.mode == 'Super_half_search':
            for m in range(len(self.operator.candidate_ops)):
                if   self.operator.candidate_ops[m].config['name']   == 'ResConvLayer':
                    for i in range(len(self.operator.candidate_ops[m].conv_block)):
                        self.operator.candidate_ops[m].conv_block[i].conv1 = conv.candidate_ops[m].conv_block[i].conv1
                        try: ## first layer 只有一个conv
                            self.operator.candidate_ops[m].conv_block[i].conv2 = conv.candidate_ops[m].conv_block[i].conv2
                        except:
                            continue

                elif   self.operator.candidate_ops[m].config['name'] == 'SpaConvLayer':
                    for i in range(len(self.operator.candidate_ops[m].conv_block)):
                        self.operator.candidate_ops[m].conv_block[i].conv1 = conv.candidate_ops[m].conv_block[i].conv1
                        self.operator.candidate_ops[m].conv_block[i].conv2 = conv.candidate_ops[m].conv_block[i].conv2

                elif self.operator.candidate_ops[m].config['name']   == 'MBInvertedConvLayer':  ## 3x3_DepthSC 和 MBInvertedConvLayer 都是MBInvertedConvLayer
                    for i in range(len(self.operator.candidate_ops[m].conv_block)):
                        self.operator.candidate_ops[m].conv_block[i].depth_conv.conv   = conv.candidate_ops[m].conv_block[i].depth_conv.conv
                        self.operator.candidate_ops[m].conv_block[i].point_linear.conv = conv.candidate_ops[m].conv_block[i].point_linear.conv


        elif  self.mode == 'Super_half_retrain':
            if self.operator.config['name']  == 'ResConvLayer':  ## SpaConvLayer 和 ResConvLayer 在config层面是一样的
                for i in range(len(self.operator.conv_block)):
                    self.operator.conv_block[i].conv1 = conv.conv_block[i].conv1
                    try:  ## first layer 只有一个conv
                        self.operator.conv_block[i].conv2 = conv.conv_block[i].conv2
                    except:
                        continue

            elif self.operator.config['name'] == 'SpaConvLayer':  ## SpaConvLayer 和 ResConvLayer 在config层面是一样的
                for i in range(len(self.operator.conv_block)):
                    self.operator.conv_block[i].conv1 = conv.conv_block[i].conv1
                    self.operator.conv_block[i].conv2 = conv.conv_block[i].conv2


            elif self.operator.config['name'] == 'MBInvertedConvLayer':  ## SpaConvLayer 和 ResConvLayer 在config层面是一样的
                for i in range(len(self.operator.conv_block)):
                    self.operator.conv_block[i].depth_conv.conv   = conv.conv_block[i].depth_conv.conv
                    self.operator.conv_block[i].point_linear.conv = conv.conv_block[i].point_linear.conv

    def forward(self, x, skip, encoder_0):
        if self.operator is None:
            # if block are abandoned across all iterations, no convs initialized
            return  x
        if self.skipped:
            # if the blocks are skipped
            return  x
        bn, c, w, h   = x.size()
        # if self.interpolate is None:
        #     self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)

        self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)
        aligned_skip  = [x] + skip
        self.skip_pre = list(set(self.skip_pre))
        use           = [(self.interpolate(aligned_skip[i])) for i in self.skip_pre]
        if encoder_0 != []:
            use.append(encoder_0)
        x             = torch.cat(use, dim=1)
        x = self.operator(x)
        return  x

    def expected_flops_Prob(self, x, skip, encoder_0):

        if self.operator is None:
            # if block are abandoned across all iterations, no convs initialized
            delta_flop = 0
            return delta_flop, x

        if self.skipped:
            # if the blocks are skipped
            delta_flop = 0
            return delta_flop, x

        bn, c, w, h  = x.size()


        self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)
        aligned_skip  = [x] + skip
        self.skip_pre = list(set(self.skip_pre))
        use = [self.interpolate(aligned_skip[i]) for i in self.skip_pre]
        if encoder_0 != []:
            use.append(encoder_0)

        # use = torch.cat(use, dim=-1)
        # x   = torch.sum(use, dim=-1, keepdim=False) / np.shape(use)[-1]

        x             = torch.cat(use, dim=1)

        expected_flops = 0
        probs_over_ops = self.operator.current_prob_over_ops
        for m, op in enumerate(self.operator.candidate_ops):
            if op is None or op.is_zero_layer():
                continue
            op_flops, x_out = op.get_flops(x)
            expected_flops += op_flops * probs_over_ops[m]

        # print('probs_over_ops: ',probs_over_ops)
        # print('expected_flops: ',expected_flops)
        return expected_flops, x_out



    def Block_flops(self, x, skip, encoder_0):
        if self.operator is None:
            # if block are abandoned across all iterations, no convs initialized
            delta_flop = 0
            params = 0

            return delta_flop, params, x

        if self.skipped:
            # if the blocks are skipped
            delta_flop = 0
            params = 0

            return delta_flop, params, x

        bn, c, w, h   = x.size()

        # if self.interpolate is None:
        #     self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)

        self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)
        aligned_skip  = [x] + skip
        self.skip_pre = list(set(self.skip_pre))
        use           = [self.interpolate(aligned_skip[i]) for i in self.skip_pre]
        if encoder_0 != []:
            use.append(encoder_0)


        x             = torch.cat(use, dim=1)
        delta_flop, parames, x = self.operator.get_flops(x)

        return delta_flop, parames, x

    def Block_flops_multi(self, x, skip, encoder_0, Block_OP_total, key_name, conv_name, Final_conv_name, iteration, layer):
        if self.operator is None:
            # if block are abandoned across all iterations, no convs initialized
            return  x

        if self.skipped:
            # if the blocks are skipped
            return  x

        bn, c, w, h   = x.size()


        self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)
        aligned_skip     = [x] + skip
        self.skip_pre    = list(set(self.skip_pre))
        use              = [self.interpolate(aligned_skip[i])  for i in self.skip_pre]
        if encoder_0 != []:
            use.append(encoder_0)

        # use             = torch.cat(use, dim=-1)
        # x               = torch.sum(use, dim=-1, keepdim=False) / np.shape(use)[-1]

        x             = torch.cat(use, dim=1)

        B_i, C_i, H_i, W_i = np.shape(x)
        flops_list, x       = self.operator.get_flops_all(x)
        B_o, C_o, H_o, W_o = np.shape(x)

        try:
            Block_OP_total[self.operator.mobile_inverted_conv._get_name() + "_" + str(iteration)+'_'+str(layer)] = flops_list
        except:
            Block_OP_total[self.operator._get_name() + "_" + str(iteration)+'_'+str(layer)] = flops_list


        for m in range(len(key_name)):
            Final_conv_name[(key_name[m].split('_')[1][:-1]  + str(iteration)+'_'+str(layer)+'_'+str(m))] = \
                (key_name[m].split('_')[1][:-1]) + '-input:' + \
                str(H_i)   + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                str(H_o)   + 'x' + str(W_o) + 'x' + str(C_o) + \
                "-expand:" + (str(conv_name[key_name[m]].split('expand:')[1].split(',')[0])) + \
                '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                '-stride:' + str(1) + \
                '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0]) + ',' \
                'value:'   + str(flops_list[m])
        return x

    def Block_params_multi(self, x, skip, encoder_0, Block_OP_total, key_name, conv_name, Final_conv_name, iteration, layer):
        if self.operator is None:
            # if block are abandoned across all iterations, no convs initialized
            return  x

        if self.skipped:
            # if the blocks are skipped
            return  x

        bn, c, w, h   = x.size()


        self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)
        aligned_skip     = [x] + skip
        self.skip_pre    = list(set(self.skip_pre))
        use              = [self.interpolate(aligned_skip[i])  for i in self.skip_pre]
        if encoder_0 != []:
            use.append(encoder_0)

        # use             = torch.cat(use, dim=-1)
        # x               = torch.sum(use, dim=-1, keepdim=False) / np.shape(use)[-1]

        x             = torch.cat(use, dim=1)

        B_i, C_i, H_i, W_i = np.shape(x)
        flops_list, x       = self.operator.get_params_all(x)
        B_o, C_o, H_o, W_o = np.shape(x)

        try:
            Block_OP_total[self.operator.mobile_inverted_conv._get_name() + "_" + str(iteration)+'_'+str(layer)] = flops_list
        except:
            Block_OP_total[self.operator._get_name() + "_" + str(iteration)+'_'+str(layer)] = flops_list


        for m in range(len(key_name)):
            Final_conv_name[(key_name[m].split('_')[1][:-1]  + str(iteration)+'_'+str(layer)+'_'+str(m))] = \
                (key_name[m].split('_')[1][:-1]) + '-input:' + \
                str(H_i)   + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                str(H_o)   + 'x' + str(W_o) + 'x' + str(C_o) + \
                "-expand:" + (str(conv_name[key_name[m]].split('expand:')[1].split(',')[0])) + \
                '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                '-stride:' + str(1) + \
                '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0]) + ',' \
                'value:'   + str(flops_list[m])
        return x

    def Block_latency(self, x, skip, encoder_0, Block_OP_total, key_name, conv_name, Final_conv_name, iteration, layer):
        if self.operator is None:
            # if block are abandoned across all iterations, no convs initialized
            return  x

        if self.skipped:
            # if the blocks are skipped
            return  x

        bn, c, w, h   = x.size()


        self.interpolate = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=True)
        aligned_skip  = [x] + skip
        self.skip_pre    = list(set(self.skip_pre))
        use           = [self.interpolate(aligned_skip[i]) for i in self.skip_pre]
        if encoder_0 != []:
            use.append(encoder_0)

        # use             = torch.cat(use, dim=-1)
        # x               = torch.sum(use, dim=-1, keepdim=False) / np.shape(use)[-1]

        x             = torch.cat(use, dim=1)

        B_i, C_i, H_i, W_i = np.shape(x)
        time_list, x       = self.operator.get_latency(x)
        B_o, C_o, H_o, W_o = np.shape(x)

        try:
            Block_OP_total[self.operator.mobile_inverted_conv._get_name() + "_" + str(iteration)+'_'+str(layer)] = time_list
        except:
            Block_OP_total[self.operator._get_name() + "_" + str(iteration)+'_'+str(layer)] = time_list


        for m in range(len(key_name)):
            Final_conv_name[(key_name[m].split('_')[1][:-1]  + str(iteration)+'_'+str(layer)+'_'+str(m))] = \
                (key_name[m].split('_')[1][:-1]) + '-input:' + \
                str(H_i)   + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                str(H_o)   + 'x' + str(W_o) + 'x' + str(C_o) + \
                "-expand:" + (str(conv_name[key_name[m]].split('expand:')[1].split(',')[0])) + \
                '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                '-stride:' + str(1) + \
                '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0]) + ',' \
                'value:'   + str(time_list[m])
        return x

    @property
    def config(self):
        return {
            'name': self.operator.__name__,
            'in_channels':  self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size':  self.kernel_size,
            'stride':       self.stride,
            'num_blocks':   self.num_blocks,
        }

class SuperProxylessNASNets(ProxylessNASNets):

    def __init__(self, conv_type, iterations, backbone, in_channel, conv_candidates,
                 n_classes=1, channel_num=None, model_name=None, gene=None, add_decoder=None, add_encoder0=None, more_down=None):
        self._redundant_modules = None
        self._unused_modules    = None
        self.num_classes        = n_classes
        self.model_name         = model_name
        self.iterations         = iterations
        self.backbone           = backbone
        self.conv_type          = conv_type
        self.stride             = 1
        self.more_down          = more_down
        self.in_channel         = in_channel

        self.num_blocks, self.filters_list = self.load_param(self.backbone, channel_num, self.iterations)

        count             = 0
        count_all         = 0
        gene              = decode(gene, self.iterations)
        add_decoder       = add_decoder
        add_encoder0      = add_encoder0
        skip_codes        = BFS(self.iterations, gene, add_decoder, add_encoder0)  # check if a block is skipped or not

        print('gene： ',      gene)
        print('skip_codes： ',skip_codes)

        enc_skip_or_not       = [[True for _ in range(self.iterations - m)] for m in range(self.iterations)]
        input_channel_assign  = self.channel_assign(skip_codes, self.filters_list, gene)
        for layer in range(self.iterations):
            for iteration in range(self.iterations - layer):
                if not skip_codes[iteration][layer]:  # if not skipped
                    enc_skip_or_not[layer] = [False]
                    continue

        enc_convs         = []
        enc_convs_all     = []
        self.encoders     = []
        for iteration in range(self.iterations):
            # initialize encoders
            for layer in range(self.iterations - iteration):
                if (iteration   == 0 and layer == 0):
                    input_channel = self.in_channel
                elif (iteration == 0 and layer != 0):
                    input_channel = input_channel_assign[iteration][layer - 1]
                elif iteration  != 0:
                    input_channel = input_channel_assign[iteration][layer]  ##@@@@
                out_channel  = self.filters_list[layer]
                if iteration == 0:
                    enc_convs.append(MixedEdge(candidate_ops=build_candidate_ops(conv_candidates if len(conv_candidates)<15 else conv_candidates['iter_'+str(count_all)], input_channel,
                                           out_channel, self.stride, 'weight_bn_act', self.num_blocks[iteration][layer] if iteration==0 else 1, group=1, layer='middle')))  ### 这样写所有网络都没有first layer
                enc_convs_all.append(MixedEdge(candidate_ops=build_candidate_ops(conv_candidates if len(conv_candidates)<15 else conv_candidates['iter_'+str(count_all)], input_channel,
                                           out_channel, self.stride, 'weight_bn_act', self.num_blocks[iteration][layer] if iteration==0 else 1, group=1, layer='middle')))
                conv_block     = enc_convs[layer]
                conv_block_all = enc_convs_all[count_all]
                block          = Block(input_channel, out_channel, conv_block, self.conv_type, gene[count][layer], mode=self.model_name, conv_blocks_all=conv_block_all)
                self.encoders.append(block)
                count_all += 1
            count += 1

        # post_transform_conv_block
        post_transform_conv_block = ConvLayer(self.filters_list[0], self.num_classes, kernel_size=1, use_bn=False, bias=True, act_func=None, ops_order='weight_bn_act', layer='final')

        self.reset_gene(gene, skip_codes)

        super(SuperProxylessNASNets, self).__init__(self.encoders, post_transform_conv_block, model_name, self.iterations, add_decoder, add_encoder0, self.more_down)

    def channel_assign(self, skip_codes, filters_list, gene):
        channel = [[True for _ in range(self.iterations - m)] for m in range(self.iterations)]
        for iter in range(self.iterations):
            for layer in range(self.iterations - iter):
                if iter == 0:
                    channel[iter][layer] = filters_list[layer]
                elif skip_codes[iter][layer]:
                    # channel[iter][layer] = None
                    channel[iter][layer] = 1

                elif not skip_codes[iter][layer]:
                    input_channel = 0
                    for i in list(set(gene[iter][layer])):
                        if i > 0:
                            input_channel += filters_list[i - 1]
                        elif i == 0:
                            input_channel += filters_list[i] if layer == 0 else filters_list[layer - 1]
                    channel[iter][layer] = input_channel

        candidate_skip_layer = [i for i in range(self.iterations - 1)]

        for iter in range(self.iterations):
            for layer in range(self.iterations - iter):
                if (iter != 0 and not skip_codes[iter][layer] and layer in candidate_skip_layer):
                    channel[iter][layer] += channel[0][layer]
                    candidate_skip_layer.remove(layer)

        return channel

    def load_param(self, backbone, channel_size, iter):

        if   backbone == 'resnet_10':
            num_blocks = [1, 1, 1, 1, 1]
        elif backbone == 'resnet_18':
            num_blocks = [1, 2, 2, 2, 2]
        elif backbone == 'resnet_34':
            num_blocks = [1, 3, 4, 6, 3]

        num_blocks = [[num_blocks[layer] for layer in range(iter - m)] for m in range(iter)]  ## pre-trans的卷积是多余的，后面记得去掉。下采样次数等于应该 iter-1

        if    channel_size == 'one':
            nb_filter = [4, 8, 16, 32, 64]
        elif  channel_size == 'two':
            nb_filter = [8, 16, 32, 64, 128]
        elif  channel_size == 'three':
            nb_filter = [16, 32, 64, 128, 256]
        elif  channel_size == 'four':
            nb_filter = [32, 64, 128, 256, 512]

        elif  channel_size == 'all_16':
            nb_filter = [16, 16, 16, 16, 16]
        elif  channel_size == 'all_32':
            nb_filter = [32, 32, 32, 32, 32]
        elif  channel_size == 'all_48':
            nb_filter = [48, 48, 48, 48, 48]

        return  num_blocks, nb_filter

    @property
    def config(self):
        raise ValueError('not needed')

    def reset_gene(self, genes, skip_genes=None):

        count = 0
        for idx, gene in enumerate(genes):
            for level in range(self.iterations - idx):
                self.encoders[count].skip_pre = gene[level]
                # print(gene[level])
                self.encoders[count].skipped = (skip_genes is not None) and skip_genes[idx][level]
                # print(skip_genes[idx][level])
                if self.encoders[count].skipped:
                    print('find one encoder skipped! at iteration ' + str(idx) + ' level ' + str(level))
                count += 1

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy        = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if   init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i]          = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, SuperProxylessNASNets)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index   = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_cpu_gpu_latency(self, run_manager, latency_model: LatencyEstimator, data_shape, device_type, ):

        expected_latency = 0
        # first conv
        expected_latency += latency_model.predict('ConvLayer_First_conv')
        # post_conv
        expected_latency += latency_model.predict('ConvLayer_Post_conv')  # 1000)
        initial_iteration = self.iterations


        # blocks
        for i in range(len(self.encoders)):
            if not self.encoders[i].skipped:
                iteration, layer =  iter_layer_compute(initial_iteration,  i)
                mb_conv          = self.encoders[i].operator

                probs_over_ops   = mb_conv.current_prob_over_ops
                for m, op in enumerate(mb_conv.candidate_ops):   ### 如果candidate有20个不变的话，LR很低的结果就是最后flops只是1/20左右，因为理想情况下是概率高的接近1
                    # if op is None or op.is_zero_layer():
                    #     continue
                    op_name          = mb_conv._get_name()+"_"+str(iteration)+"_"+str(layer)
                    op_latency       = latency_model.predict(op_name, id=m)
                    expected_latency = expected_latency + op_latency * probs_over_ops[m]

        return expected_latency*1000


    def Onehot_cpu_gpu_latency(self, run_manager, latency_model: LatencyEstimator ):  ### 没有像上面用onehot
        expected_latency = 0
        expected_latency += latency_model.predict('ConvLayer_First_conv')
        expected_latency += latency_model.predict('ConvLayer_Post_conv')  # 1000)
        initial_iteration = self.iterations
        for i in range(len(self.encoders)):
            if not self.encoders[i].skipped:
                iteration, layer =  iter_layer_compute(initial_iteration,  i)
                mb_conv          = self.encoders[i].operator
                for m, op in enumerate(mb_conv.candidate_ops):   ### 如果candidate有20个不变的话，LR很低的结果就是最后flops只是1/20左右，因为理想情况下是概率高的接近1
                    if op is None or op.is_zero_layer():
                        continue
                    op_name          = mb_conv._get_name()+"_"+str(iteration)+"_"+str(layer)
                    op_latency       = latency_model.predict(op_name, id=m)
                    expected_latency = expected_latency + op_latency
        return expected_latency*1000


    def Thero_flops_params_latency(self, run_manager, latency_model: LatencyEstimator ):  ### 没有像上面用onehot
        expected_latency = 0
        expected_latency += latency_model.predict('ConvLayer_First_conv')
        expected_latency += latency_model.predict('ConvLayer_Post_conv')  # 1000)
        initial_iteration = self.iterations
        for i in range(len(self.encoders)):
            if not self.encoders[i].skipped:
                iteration, layer =  iter_layer_compute(initial_iteration,  i)
                mb_conv          = self.encoders[i].operator

                probs_over_ops   = mb_conv.current_prob_over_ops
                for m, op in enumerate(mb_conv.candidate_ops):   ### 如果candidate有20个不变的话，LR很低的结果就是最后flops只是1/20左右，因为理想情况下是概率高的接近1
                    # if op is None or op.is_zero_layer():
                    #     continue
                    op_name          = mb_conv._get_name()+"_"+str(iteration)+"_"+str(layer)
                    op_latency       = latency_model.predict(op_name, id=m)
                    expected_latency = expected_latency + op_latency * probs_over_ops[m]
        return expected_latency


    def expected_flops(self, latency_model: LatencyEstimator, ):
        expected_latency  = 0
        conv_num          = 0
        # first conv
        expected_latency += latency_model.predict('ConvLayer_First_conv')
        # post_conv
        expected_latency += latency_model.predict('ConvLayer_Post_conv')  # 1000)
        initial_iteration = self.iterations
        # blocks
        for i in range(len(self.encoders)):
            if not self.encoders[i].skipped:
                iteration, layer =  iter_layer_compute(initial_iteration,  i)
                mb_conv          = self.encoders[i].operator

                probs_over_ops   = mb_conv.current_prob_over_ops
                for m, op in enumerate(mb_conv.candidate_ops):   ### 如果candidate有20个不变的话，LR很低的结果就是最后flops只是1/20左右，因为理想情况下是概率高的接近1
                    # if op is None or op.is_zero_layer():
                    #     continue
                    op_name          = mb_conv._get_name()+"_"+str(iteration)+"_"+str(layer)
                    op_latency       = latency_model.predict(op_name, m, conv_num)
                    expected_latency = expected_latency   + op_latency * probs_over_ops[m]
                    conv_num+=1

        return expected_latency

    def expected_params(self, latency_model: LatencyEstimator, ):
        expected_latency = 0
        # first conv
        expected_latency += latency_model.predict('ConvLayer_First_conv')
        # post_conv
        expected_latency += latency_model.predict('ConvLayer_Post_conv')  # 1000)
        initial_iteration = self.iterations

        # blocks
        for i in range(len(self.encoders)):
            if not self.encoders[i].skipped:
                iteration, layer =  iter_layer_compute(initial_iteration,  i)
                mb_conv          = self.encoders[i].operator

                probs_over_ops   = mb_conv.current_prob_over_ops
                for m, op in enumerate(mb_conv.candidate_ops):   ### 如果candidate有20个不变的话，LR很低的结果就是最后flops只是1/20左右，因为理想情况下是概率高的接近1
                    # if op is None or op.is_zero_layer():
                    #     continue
                    op_name          = mb_conv._get_name()+"_"+str(iteration)+"_"+str(layer)
                    op_latency       = latency_model.predict(op_name, id=m)
                    expected_latency = expected_latency + op_latency * probs_over_ops[m]

        return expected_latency


    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return ProxylessNASNets(list(self.encoders),  self.post_transform_conv_block,  self.model_name,  self.iterations,
                                self.add_decoder,     self.add_encoder0,               self.more_down)
