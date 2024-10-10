
import os
import time
import json
from datetime import timedelta
import numpy as np
import copy

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from search.utils import *
from search.utils.utils import SoftIoULoss, mIoU, ROCMetric, PD_FA, save_Ori_intensity_Pred_GT, make_visulization_dir, load_dataset_eva
import torch.nn.functional as F
import time
import yaml
import scipy.io as scio
import shutil
from   PIL import Image, ImageOps, ImageFilter
from  tqdm import tqdm
from CRF_lib.DenseCRF import  dense_crf
from CRF_lib.imutils import  *
import torchvision.utils as utils

class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 model_init, init_div_groups, validation_frequency, print_frequency, id_mode, root, split_method, base_size, crop_size, suffix, eval_batch_size):
        self.n_epochs          = n_epochs
        self.init_lr           = init_lr
        self.lr_schedule_type  = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset           = dataset
        self.train_batch_size  = train_batch_size
        self.test_batch_size   = test_batch_size
        self.valid_size        = valid_size

        self.opt_type          = opt_type
        self.opt_param         = opt_param
        self.weight_decay      = weight_decay
        self.label_smoothing   = label_smoothing
        self.no_decay_keys     = no_decay_keys

        self.model_init           = model_init
        self.init_div_groups      = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency      = print_frequency

        self.id_mode            = id_mode
        self.root            = root
        self.split_method    = split_method
        self.base_size       = base_size
        self.crop_size       = crop_size
        self.suffix          = suffix
        self.eval_batch_size = eval_batch_size
        self._data_provider  = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None, fixed_lr=None):
        if  self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur   = epoch * nBatch + batch
            lr      = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self.lr_schedule_type == 'fixed':
            lr      = fixed_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None, fixed_lr=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch, fixed_lr)
        # new_lr = 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        raise NotImplementedError

    @property
    def data_provider(self):
        if self._data_provider is None:
            if  'SIRST' in self.dataset:
                from search.data_providers.SIRST    import SIRST_DataProvider
                self._data_provider = SIRST_DataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train_data

    @property
    def valid_loader(self):
        return self.data_provider.val_data

    @property
    def test_loader(self):
        return self.data_provider.test_data

    @property
    def inference_loader(self):
        return self.data_provider.inference_data

    @property
    def inference_loader_resize(self):
        return self.data_provider.inference_data_resize

    @property
    def inference_loader_latency(self):
        return self.data_provider.inference_data_latency

    @property
    def VisualziationLoader(self):
        return self.data_provider.VisualziationLoader


    @property
    def inference_loader_vis(self):
        return self.data_provider
    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer """

    def build_optimizer(self, net_params):

        if self.opt_type == 'Adagrad':
            optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, net_params), lr=self.init_lr)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True, crop_size=None,
                  conv_name=None, target_hardware=None, model_name=None, iterations=None, fast=None, mode=None, gene=None,
                  candidates_type=None, random_choose=None,second_train=False ):
        self.path         = path
        self.net          = net
        self.run_config   = run_config
        self.out_log      = out_log

        self._logs_path, self._save_path = None, None
        self.best_IOU        = 0
        self.start_epoch     = 0
        self.iterations      = iterations
        self.mode            = mode
        self.gene            = gene
        self.candidates_type = candidates_type
        
        # initialize model (default)
        # self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # a copy of net on cpu for latency estimation & mobile latency model
        self.net_on_cpu_for_latency = copy.deepcopy(self.net).cpu()
        self.crop_size              = crop_size

        if  torch.cuda.device_count()>1:
            ## If multi-GPU
            self.device     = torch.device('cuda:0')
            self.net        = torch.nn.DataParallel(self.net)
            self.net.to(self.device)
            cudnn.benchmark = True
        else:
            ## If single-GPU
            self.device = torch.device('cuda:0')
            self.net.to('cuda:0')

        # # net info    # compute cpu&gpu cost
        if second_train!= True:
            initial_metric,metric_txt  = self.compute_latency(target_hardware, conv_name, model_name, self.iterations, fast, random_choose)
            self.latency_estimator     = LatencyEstimator(path, initial_metric, metric_txt)
            # self.print_net_info(measure_latency, fast)

        self.criterion = SoftIoULoss
        self.optimizer = self.run_config.build_optimizer(self.net.module.weight_parameters()  if  torch.cuda.device_count()>1 else self.net.weight_parameters())



    """ save path and log path """


    def compute_latency(self, target_hardware, conv_name, model_name, iterations, fast, random_choose):
        ori_path        = self.path.split('logs/')[0] + 'logs'
        Ready_list      = os.listdir(ori_path)
        operation_name  = []
        operation_value = []
        for i in range(len(Ready_list)):
            if random_choose =='True':
                continue
            if ('search' in self.mode) and (self.gene.split('/phase1_gene.txt')[0][-8:] in Ready_list[i]) \
                                       and (target_hardware in Ready_list[i]) and ('yaml' in Ready_list[i]) and(self.candidates_type in Ready_list[i]):
                print('load_exist_yaml')

                metric_txt = {}
                initial_metric = {}
                with open(ori_path+'/'+Ready_list[i], 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                for name, value in config.items():
                    operation_name.append(name)
                    operation_value.append(value)
                cost_list = []
                iteration = 5
                layer = 0
                memory =[]
                for id in range(0, len(operation_name)):
                    metric_txt[operation_name[id]] = operation_value[id]
                    if str(5-iteration)+"_"+str(layer)+'_' in operation_name[id]:
                        cost_list.append(operation_value[id].split('value:')[1])
                    else:
                        if len(cost_list)>5:
                            initial_metric['MixedEdge_' + str(5 - iteration) + "_" + str(layer)] = cost_list
                        else:
                            memory.append(operation_value[id-1].split('value:')[1])
                        cost_list = []
                        if memory != []:
                            cost_list.append(memory[0])
                            memory = []
                        cost_list.append(operation_value[id].split('value:')[1])
                        layer += 1
                        if layer == (iteration):
                            layer = 0
                            iteration = iteration - 1
                initial_metric['ConvLayer_Post_conv'] = operation_value[-1].split('value:')[1]


        if operation_name==[]:
            if fast == 'True':
                sample_num = 1
            else:
                sample_num  = 100
            device_type           = target_hardware
            initial_metric       = {}
            initial_metric_plain = []
            Final_metric_txt     = {}


            if ('cpu' in target_hardware or 'gpu' in target_hardware):
                for i in range(sample_num):
                    if i == 0:
                        initial_metric, metric_txt, device_type  = self.comput_cpu_gpu_cost(device_type, conv_name, model_name, iterations)
                    else:
                        latency_total,   metric_txt, device_type = self.comput_cpu_gpu_cost(device_type, conv_name, model_name, iterations)
                    if i != 0:
                        for key, value in latency_total.items():
                            if key in initial_metric:  ## 遍历key中的所有value
                                for m in range(len(initial_metric[key])):
                                    initial_metric[key][m] += value[m]

            elif target_hardware=='flops' :
                sample_num = 1
                initial_metric, metric_txt, device_type = self.flops_cost(device_type, conv_name, model_name, iterations)
            elif target_hardware == 'params':
                sample_num = 1
                initial_metric, metric_txt, device_type = self.params_cost(device_type, conv_name, model_name, iterations)

            for key, value in initial_metric.items():
                for m in range(len(initial_metric[key])):
                    try:
                        initial_metric[key][m] /= sample_num
                    except:
                        print()

            for value in initial_metric.values():
                for m in range(len(value)):
                    initial_metric_plain.append(value[m])

            metric_txt_key = [key for key in metric_txt.keys()]
            for i in range(len(metric_txt_key)):
                Final_metric_txt[metric_txt_key[i]] = metric_txt[metric_txt_key[i]].split(',value:')[0]+',value:'+\
                                                                   str(initial_metric_plain[i])
            self.write_latency_txt(Final_metric_txt, device_type)

        return initial_metric,metric_txt

    def write_latency_txt(self, Final_latency_txt, device_type):
        # yaml      = YAML()
        yaml_path = self.path+'/'+ 'Latency_' + device_type + '.yaml'
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(Final_latency_txt, f)

    @property
    def save_path(self):
        if self._save_path is None:
            save_path       = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path       = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ net info """

    def comput_cpu_gpu_cost(self, measure_latency, conv_name, model_name, iterations):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)
        if  'cpu' in measure_latency:
            input_var   = torch.zeros(data_shape, device=torch.device('cpu'))
            net         = self.net_on_cpu_for_latency
            device_type = 'cpu'

        elif 'gpu' in measure_latency:
            self.device = torch.device('cuda:0')
            input_var   = torch.zeros(data_shape, device=torch.device(self.device))
            if isinstance(self.net, nn.DataParallel):
                net = self.net.module
            else:
                net = self.net
            device_type = 'gpu'

        Final_conv_name = {}
        Block_OP_total  = {}
        with torch.no_grad():
            x = input_var

            key_name = {}
            dic_value = [value.keys() for value in conv_name.values()]
            for h in range(int((1 + iterations) * iterations / 2)):
                key_name['iter_' + str(h)] = list(dic_value[h])

            e_i = 0
            this_layer = iterations
            for iteration in range(iterations):
                # print('this_layer：', this_layer)
                enc = [None for i in range(this_layer)]
                if iteration == 0:
                    encoder_0 = [None for i in range(this_layer)]
                # encoding path
                for layer in range(this_layer):
                    if (layer == 0 and iteration == 0):
                        x_in = x
                    x_in = net.encoders[e_i].Block_latency(x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (iteration != 0 and net.encoders[e_i].skipped == False) else [],
                                             Block_OP_total, key_name['iter_'+str(e_i)], conv_name['iter_'+str(e_i)], Final_conv_name, iteration, layer)
                    encoder_0[layer] = x_in if iteration == 0 else encoder_0[layer]
                    encoder_0[layer] = [] if ((iteration != 0) and net.encoders[e_i].skipped == False) else encoder_0[layer]

                    if iteration == 0:
                        encoder_0[layer] = x_in
                    enc[layer] = x_in
                    x_in = F.max_pool2d(x_in, 2)
                    e_i = e_i + 1
                this_layer -= 1
                enc_after = enc
                x_in = enc_after[0]


            ### Post process
            B_i, C_i, H_i, W_i   = np.shape(input_var)
            time_list, x         = net.post_transform_conv_block.get_latency(x_in)
            B_o, C_o, H_o, W_o   = np.shape(x)
            Block_OP_total[net.post_transform_conv_block._get_name() + "_" + 'Post_conv']=time_list
            Final_conv_name['Post_conv:'] = str(H_i)  + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + str(H_o) +'x' + str(W_o) + 'x' + str(C_o) + ',' + \
                                             'value:' + str(time_list[0])

            return Block_OP_total, Final_conv_name, device_type

    def flops_cost(self, measure_latency, conv_name, model_name, iterations):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        input_var  = torch.zeros(data_shape, device=torch.device(self.device))
        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net

        Final_conv_name = {}
        Block_OP_total  = {}
        with torch.no_grad():
            x = input_var
            # key_name = [keys for keys in conv_name.keys()]

            key_name  = {}
            dic_value = [value.keys() for value in conv_name.values()]
            for h in range(int((1+iterations)*iterations/2)):
                key_name['iter_'+str(h)] = list(dic_value[h])

            e_i = 0
            this_layer = iterations
            for iteration in range(iterations):
                # print('this_layer：', this_layer)
                enc = [None for i in range(this_layer)]
                if iteration == 0:
                    encoder_0 = [None for i in range(this_layer)]
                # encoding path
                for layer in range(this_layer):
                    if (layer == 0 and iteration == 0):
                        x_in = x
                    # x_in = net.encoders[e_i].Block_flops_multi(x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (
                    #             layer == (this_layer - 1) and iteration != 0) else [], Block_OP_total, key_name, conv_name, Final_conv_name, iteration, layer)

                    x_in = net.encoders[e_i].Block_flops_multi(x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (iteration != 0 and net.encoders[e_i].skipped == False) else [],
                                             Block_OP_total, key_name['iter_'+str(e_i)], conv_name['iter_'+str(e_i)], Final_conv_name, iteration, layer)
                    encoder_0[layer] = x_in if iteration == 0 else encoder_0[layer]
                    encoder_0[layer] = []   if ((iteration != 0) and net.encoders[e_i].skipped == False) else encoder_0[layer]

                    if iteration == 0:
                        encoder_0[layer] = x_in
                    enc[layer] = x_in
                    x_in = F.max_pool2d(x_in, 2)
                    e_i = e_i + 1
                this_layer -= 1
                enc_after = enc
                x_in = enc_after[0]

            ### Post process
            B_i, C_i, H_i, W_i   = np.shape(x_in)
            flop_list, x         = net.post_transform_conv_block.get_flops_all(x_in)
            B_o, C_o, H_o, W_o   = np.shape(x)
            Block_OP_total[net.post_transform_conv_block._get_name() + "_" + 'Post_conv']=flop_list
            Final_conv_name['Post_conv:'] = str(H_i)  + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + str(H_o) +'x' + str(W_o) + 'x' + str(C_o) + ',' + \
                                             'value:' + str(flop_list[0])

            return Block_OP_total, Final_conv_name, measure_latency

    def params_cost(self, measure_latency, conv_name, model_name, iterations):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        input_var  = torch.zeros(data_shape, device=torch.device(self.device))
        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net

        Final_conv_name = {}
        Block_OP_total  = {}
        with torch.no_grad():
            x = input_var
            key_name = {}
            dic_value = [value.keys() for value in conv_name.values()]
            for h in range(int((1 + iterations) * iterations / 2)):
                key_name['iter_' + str(h)] = list(dic_value[h])

            e_i = 0
            this_layer = iterations
            for iteration in range(iterations):
                # print('this_layer：', this_layer)
                enc = [None for i in range(this_layer)]
                if iteration == 0:
                    encoder_0 = [None for i in range(this_layer)]
                # encoding path
                for layer in range(this_layer):
                    if (layer == 0 and iteration == 0):
                        x_in = x
                    # x_in = net.encoders[e_i].Block_flops_multi(x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (
                    #             layer == (this_layer - 1) and iteration != 0) else [], Block_OP_total, key_name, conv_name, Final_conv_name, iteration, layer)

                    x_in = net.encoders[e_i].Block_params_multi(x_in, enc_after if iteration != 0 else [], encoder_0[layer] if (iteration != 0 and net.encoders[e_i].skipped == False) else [],
                                             Block_OP_total, key_name['iter_'+str(e_i)], conv_name['iter_'+str(e_i)], Final_conv_name, iteration, layer)
                    encoder_0[layer] = x_in if iteration == 0 else encoder_0[layer]
                    encoder_0[layer] = []   if ((iteration != 0) and net.encoders[e_i].skipped == False) else encoder_0[layer]

                    if iteration == 0:
                        encoder_0[layer] = x_in
                    enc[layer] = x_in
                    x_in = F.max_pool2d(x_in, 2)
                    e_i = e_i + 1
                this_layer -= 1
                enc_after = enc
                x_in = enc_after[0]

            ### Post process
            B_i, C_i, H_i, W_i   = np.shape(x_in)
            flop_list, x         = net.post_transform_conv_block.get_flops_all(x_in)
            B_o, C_o, H_o, W_o   = np.shape(x)
            Block_OP_total[net.post_transform_conv_block._get_name() + "_" + 'Post_conv']=flop_list
            Final_conv_name['Post_conv:'] = str(H_i)  + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + str(H_o) +'x' + str(W_o) + 'x' + str(C_o) + ',' + \
                                             'value:' + str(flop_list[0])

            return Block_OP_total, Final_conv_name, measure_latency
    # noinspection PyUnresolvedReferences
    def net_flops(self):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net
        input_var  = torch.zeros(data_shape, device=self.device)
        with torch.no_grad():
            flop, total_params, _ = net.cuda().get_flops(input_var)
        return flop, total_params

    # noinspection PyUnresolvedReferences
    def net_latency(self, l_type='gpu4', fast='True', given_net=None):  ##后面的数字代表batch size是多少


        if 'gpu' in l_type:
            l_type, batch_size = l_type[:3], 1
        else:
            batch_size = 1
        data_shape = [batch_size] + list(self.run_config.data_provider.data_shape)

        if given_net is not None:
            net = given_net
        else:
            if torch.cuda.device_count()>1:
                net = self.net.module
            else:
                net = self.net

        if l_type == 'mobile':
            predicted_latency = 200
            print('fail to predict the mobile latency')
            return predicted_latency, None

        elif l_type == 'cpu':
            if fast == 'True':
                n_warmup = 1
                n_sample = 2
            else:
                n_warmup = 10
                n_sample = 100
            try:
                self.net_on_cpu_for_latency.set_active_via_net(net)
            except AttributeError:
                print(type(self.net_on_cpu_for_latency), ' do not `support set_active_via_net()`')
            net    = self.net_on_cpu_for_latency
            images = torch.zeros(data_shape, device=torch.device('cpu'))

        elif l_type == 'gpu':
            if fast == 'True':
                n_warmup = 5
                n_sample = 10
            else:
                n_warmup = 50
                n_sample = 100
            images = torch.zeros(data_shape, device=self.device)
        else:
            raise NotImplementedError

        measured_latency = {'warmup': [], 'sample': []}
        net.eval()
        with torch.no_grad():
            for i in range(n_warmup + n_sample):
                start_time = time.time()
                net(images)
                used_time  = (time.time() - start_time) * 1e3  # ms
                if i >= n_warmup:
                    measured_latency['sample'].append(used_time)
                else:
                    measured_latency['warmup'].append(used_time)
        net.train()
        return sum(measured_latency['sample']) / n_sample, measured_latency

    def print_net_info(self, measure_latency=None, fast=None):
        # parameters
        if isinstance(self.net, nn.DataParallel):
            total_params = count_parameters(self.net.module)
        else:
            total_params = count_parameters(self.net)
        if self.out_log:
            print('Total training params: %.2fM' % (total_params / 1e6))
        net_info = {'param': '%.2fM' % (total_params / 1e6),}

        # flops
        flops = self.net_flops()
        if self.out_log:
            print('Total FLOPs: %.1fM' % (flops / 1e6))
        net_info['flops'] = '%.1fM' % (flops / 1e6)

        # latency
        latency_types = [] if measure_latency is None else measure_latency.split('#')
        for l_type in latency_types:
            latency, measured_latency = self.net_latency(l_type, fast=fast, given_net=None)
            if self.out_log:
                print('Estimated %s latency: %.3fms' % (l_type, latency))
            net_info['%s latency' % l_type] = {
                'val': latency,
                'hist': measured_latency
            }
        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict() if torch.cuda.device_count()>1 else self.net.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path   = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint  = torch.load(model_fname)
            else:
                checkpoint  = torch.load(model_fname, map_location='cpu')

            self.net.module.load_state_dict(checkpoint['state_dict'])
            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc    = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.save_config() if torch.cuda.device_count()>1 else self.net.save_config(), open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def validate(self, is_test=True, use_train_mode=False):

        data_loader = self.run_config.test_loader
        net         = self.net
        self.mIoU   = mIoU(1)
        self.mIoU.reset()

        # if net is None:
        #     net = self.net
        if use_train_mode:
            net.train()
        else:
            net.eval()
        batch_time = AverageMeter()
        losses     = AverageMeter()
        end        = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                loss   = SoftIoULoss(output, labels)

                # measure accuracy and record loss
                losses.update(loss, images.size(0))
                self.mIoU.update(output, labels)
                _, validate_IoU = self.mIoU.get()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                        'validate_IoU {validate_IoU:.3f}'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, validate_IoU=validate_IoU) ## 括号外面是当前batch的结果，里面的是平均的结果
                    print(test_log)

            return losses.avg, validate_IoU

    def inference(self, args, is_test=True, use_train_mode=False):

        if   args.Inference_resize == 'False':
             data_loader = self.run_config.inference_loader
        elif args.Inference_resize == 'True':
             data_loader = self.run_config.inference_loader_resize
        tbar = tqdm(data_loader)

        net         = self.net
        self.mIoU   = mIoU(1)
        self.ROC    = ROCMetric(1, args.ROC_thr)
        self.PD_FA  = PD_FA(1,10, args.crop_size)

        target_image_path = args.path + '/' + 'visulization_result'
        target_dir        = args.path + '/' + 'visulization_fuse'

        train_img_ids, val_img_ids, test_txt = load_dataset_eva(args.root, args.dataset, args.split_method)

        make_visulization_dir(target_image_path, target_dir)


        if use_train_mode:
            net.train()
        else:
            net.eval()
        batch_time = AverageMeter()
        losses     = AverageMeter()
        end        = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            num = 0
            for i, (images, labels, ori_img, img_id) in enumerate(tbar):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output,_ = net(images)

                # if 'Misc_111' in img_id:
                #     labels = labels[:,:,:,:,0]
                loss   = SoftIoULoss(output, labels)


                if args.postprocess == 'crf':
                    MODEL_NUM_CLASSES = 2
                    CRF_ITER = 1

                    Target_probability_map     = (np.array((output > 0).cpu()).astype('int64')*np.array((output).cpu())/np.array((output).cpu()).max())[0]
                    Background_probability_map = 1 - Target_probability_map
                    prob     = np.concatenate([Background_probability_map, Target_probability_map], axis=0)
                    prob     = dense_crf(prob, np.transpose(np.array(ori_img.cpu()[0]), (2,0,1)), n_classes=MODEL_NUM_CLASSES, n_iters=CRF_ITER)
                    prob_seg = prob.astype(np.float32)
                    output   = torch.from_numpy(np.argmax(prob_seg, axis=0)).type(torch.float32).cuda()
                    output   = torch.unsqueeze(torch.unsqueeze(output, 0),0)
                save_Ori_intensity_Pred_GT(output, labels, target_image_path, val_img_ids, num, args.suffix, output.size()[2], output.size()[3])

                num += 1
                self.ROC.  update(output, labels)
                self.mIoU. update(output, labels)
                self.PD_FA.update(output, labels)

                losses.    update(loss.item(), output.size(0))
                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids), args.crop_size)
            test_loss = losses.avg

            with open(args.path + '/' + 'Pd_Fa.txt', 'a') as  f:
                f.write('PD:')
                f.write("\t")
                for i in range(len(PD)):
                    f.write(str(PD[i]))
                    f.write("\t")
                f.write("\n")

                f.write('FA:')
                f.write("\t")
                for i in range(len(FA)):
                    f.write(str(FA[i]))
                    f.write("\t")

            scio.savemat(args.path + '/' + '/' +'PD_FA_' + str(255),
                         {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('PD:',PD)
            print('FA:',FA)
            self.best_iou = mean_IOU

            source_image_path = args.root +'/' + args.dataset + '/images'

            txt_path = test_txt
            ids = []
            with open(txt_path, 'r') as f:
                ids += [line.strip() for line in f.readlines()]

            for i in range(len(ids)):
                source_image = source_image_path + '/' + ids[i] + args.suffix
                target_image = target_image_path + '/' + ids[i] + args.suffix
                shutil.copy(source_image, target_image)
            # for i in range(len(ids)):
            #     source_image = target_image_path + '/' + ids[i] + args.suffix
            #     img = Image.open(source_image)
            #     img = img.resize((args.crop_size, args.crop_size), Image.ANTIALIAS)
            #     img.save(source_image)

    def inference_latency(self, args, is_test=True, use_train_mode=False):
        net  = self.net
        data_loader = self.run_config.inference_loader_latency
        tbar = tqdm(data_loader)
        if use_train_mode:
            net.train()
        else:
            net.eval()
        with torch.no_grad():
            for i, (images, labels, ori_img, img_id) in enumerate(tbar):
                images, labels = images.to(self.device), labels.to(self.device)
                start_time     =  time.time()
                for h in range(args.Inference_repeated):
                    output = net(images)
                end_time   = time.time()
                gpu_avg_time   = (end_time-start_time)/args.Inference_repeated

            experiment_num = args.Inference_repeated
            for i, (images, labels, ori_img, img_id) in enumerate(tbar):
                images, labels = images.to(self.device), labels.to(self.device)
                start_time     =  time.time()
                for h in range(experiment_num):
                    output = net(images)
                    end_time   = time.time()
                    print('avg_time:', (end_time - start_time) / (h + 1))

        with torch.no_grad():
            for i, (images, labels, ori_img, img_id) in enumerate(tbar):
                start_time     =  time.time()
                for h in range(args.Inference_repeated):
                    output = net.cpu()(images)
                end_time   = time.time()
                cpu_avg_time   = (end_time-start_time)/args.Inference_repeated

            experiment_num = args.Inference_repeated
            for i, (images, labels, ori_img, img_id) in enumerate(tbar):
                start_time     =  time.time()
                for h in range(experiment_num):
                    output = net.cpu()(images)
                    end_time   = time.time()
                    print('avg_time:', (end_time - start_time) / (h + 1))

        return gpu_avg_time, cpu_avg_time


    def visualization (self, args, is_test=True, model_name=None, use_train_mode=False):
        net  = self.net
        data_loader = self.run_config.VisualziationLoader
        tbar = tqdm(data_loader)
        if use_train_mode:
            net.train()
        else:
            net.eval()
        with torch.no_grad():
            for i, (images, ori_img, img_id) in enumerate(tbar):
                images = images.to(self.device)
                output, vis_output = net(images)

                feature_layer = ['x4_0', 'x3_1', 'x2_2', 'x1_3', 'x0_4']
                for m in range(len(feature_layer)):
                    feature = vis_output[m]
                    N1, C1, W1, H1 = feature.size()
                    feature = feature.sum(dim=1).view(1, 1, W1, H1)
                    up_factor = 256 / W1
                    feature = F.interpolate(feature, scale_factor=up_factor, mode='bilinear', align_corners=False)
                    attn = utils.make_grid(feature, nrow=1, normalize=True, scale_each=True)
                    attn = attn.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
                    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
                    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
                    print('shape(attn):', np.shape(attn))
                    img_out = np.array(ori_img)
                    # vis = 0.3* img_out + 0.7 * attn
                    vis = attn
                    print(np.shape(vis))
                    img = Image.fromarray(np.uint8(vis))
                    save_dir = img_id[0].split('.png')[0] +'_' + model_name + '_' + feature_layer[m] + '.jpg'
                    img.save(save_dir)

        return



    def train_one_epoch(self, adjust_lr_func, train_log_func):
        from torchstat import stat

        self.mIoU  = mIoU(1)
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses     = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr         = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)

            # compute output
            output   = self.net(images)
            # stat         = stat(self.net.cpu(), (3, 256, 256))

            loss     = SoftIoULoss(output, labels)

            # measure accuracy and record loss
            losses.update(loss,  images.size(0))
            self.mIoU.update(output, labels)
            _, train_IoU = self.mIoU.get()

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.run_config.print_frequency == 0 or i + 1 == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, train_IoU, new_lr)
                self.write_log(batch_log, 'train')
        return train_IoU

    def train(self, fixed_lr=None):
        nBatch = len(self.run_config.train_loader)

        def train_log_func(epoch_, i, batch_time, data_time, losses, train_IoU, lr):
            batch_log  = 'Train [{0}][{1}/{2}]\t' \
                         'Time  {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data  {data_time.val:.3f}  ({data_time.avg:.3f})\t' \
                         'Loss  {losses.val:.4f}     ({losses.avg:.4f})\t' \
                         'train_IoU {train_IoU:.3f}'. \
                         format(epoch_ + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time, losses=losses, train_IoU=train_IoU)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

            end       = time.time()
            train_IoU = self.train_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, nBatch, fixed_lr=fixed_lr),
                lambda i, batch_time, data_time, losses, train_IoU, new_lr: train_log_func(epoch, i, batch_time, data_time, losses, train_IoU, new_lr))

            time_per_epoch = time.time() - end
            seconds_left   = int((self.run_config.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))


            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, validate_IoU = self.validate(is_test=False)
                is_best       = validate_IoU > self.best_IOU
                self.best_IOU = max(self.best_IOU, validate_IoU)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\tValidate_IoU {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.run_config.n_epochs, val_loss, validate_IoU, self.best_IOU)

                self.write_log(val_log, 'valid')
            else:
                is_best = False

            self.save_model({
                'epoch':      epoch,
                'best_IOU':   self.best_IOU,
                'optimizer':  self.optimizer.state_dict(),
                'state_dict': self.net.module.state_dict() if torch.cuda.device_count()>1 else self.net.state_dict(),
            }, is_best=is_best)
