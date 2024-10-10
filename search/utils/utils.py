from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from skimage import measure

from torch.nn import init
from datetime import datetime
import argparse
import shutil
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# matplotlib.use('TkAgg')


def conv_name_define(conv_candidates, iterations):
    conv_name       = {}
    conv_name_final = {}

    for m in range(int((1+iterations)*iterations/2)):
        conv_candidates_list = conv_candidates if len(conv_candidates) <16 else conv_candidates['iter_'+str(m)]
        for i in range(len(conv_candidates_list)):
            expand  = conv_candidates_list[i].split('_MBConv')[1]    if '_MBConv'    in conv_candidates_list[i] else 0
            kernel  = conv_candidates_list[i].split('x')[0]
            group   = conv_candidates_list[i].split('_GroupConv')[1] if '_GroupConv' in conv_candidates_list[i] else 0
            conv_name[conv_candidates_list[i]] = 'expand:' + str(expand) + "," + 'kernel:' + str(kernel) + "," + 'group:' + str(group)+ ","
        conv_name_final['iter_'+str(m)] = conv_name
        conv_name ={}
    return conv_name_final

def middle_block_latency_encoder(x, net, i, Block_OP_total, key_name, conv_name, Final_conv_name):

    B_i, C_i, H_i, W_i = np.shape(x)
    time_list, x       = net.blocks[i].get_latency(x)
    B_o, C_o, H_o, W_o = np.shape(x)
    try:
        Block_OP_total[net.blocks[i].mobile_inverted_conv._get_name() + "_" + str(i)] = time_list
    except:
        Block_OP_total[net.blocks[i]._get_name() + "_" + str(i)] = time_list
    if i != int(len(net.blocks) / 2):
        x = F.max_pool2d(x, 2)

    for m in range(len(key_name)):
        Final_conv_name[(key_name[m].split('_')[1][:-1] + str(i) + str(m)) if i!=0 else 'block1'] = \
                                                                            (key_name[m].split('_')[1][:-1] if i!=0 else (net.blocks[0].candidate_ops[0]._get_name())) + '-input:' + \
                                                                            str(H_i) + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                                                                            str(H_o) + 'x' + str(W_o) + 'x' + str(C_o) + \
                                                                            "-expand:" + (str(conv_name[key_name[m]].split('expand:')[1].split(',')[0]) if i != 0 else str(1)) + \
                                                                            '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                                                                            '-stride:' + str(1) + \
                                                                            '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0]) + ',' \
                                                                              'value:' + str(time_list[m])
        if i == 0:  ### conv-0 只有一个Block
            break
    return  x


def random_operation_choose(candidates_type, iterations, random_choose):
    conv_candidates_final = {}
    conv_num              = (1+iterations)*iterations/2

    # if candidates_type=='MBInverted':
    #     conv_candidates = ['3x3_MBConv2',    '3x3_MBConv4',    '3x3_MBConv6']
    # else:
    #     conv_candidates       =    ['3x3_MBConv2',    '3x3_MBConv4',    '3x3_MBConv6',
    #                                 '3x3_ResConv',    '5x5_ResConv',    '7x7_ResConv',
    #                                 '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8',
    #                                 '3x3_SpaConv',    '5x5_SpaConv',    '7x7_SpaConv']
    # for i in range(int(conv_num)):
    #     conv_candidates_final['iter'+'_'+str(i)] = random.sample(conv_candidates, len(conv_candidates) if random_choose =='False' else int(random_choose/2))
    # return  conv_candidates_final


    conv_candidates_performance = ['3x3_ResConv',    '5x5_ResConv',    '7x7_ResConv',
                                   '3x3_GroupConv2',
                                   '3x3_SpaConv',    '5x5_SpaConv',    '7x7_SpaConv']
    conv_candidates_efficiency  = ['3x3_MBConv2',
                                   '3x3_ResConv',
                                   '3x3_GroupConv2','3x3_GroupConv4', '3x3_GroupConv8',
                                   '3x3_SpaConv']
    for i in range(int(conv_num)):
        if i <5:
            conv_candidates_final['iter'+'_'+str(i)] = random.sample(conv_candidates_performance, len(conv_candidates_performance) if random_choose =='False' else int(len(random_choose)/2))
        else:
            conv_candidates_final['iter'+'_'+str(i)] = random.sample(conv_candidates_efficiency, len(conv_candidates_efficiency) if random_choose =='False' else int(len(random_choose)/2))

    return  conv_candidates_final


def operation_choose(candidates_type):
    # if args.candidates_type   == 'whole':
    #     args.conv_candidates  = [ '3x3_DepthSC',    '5x5_DepthSC',    '7x7_DepthSC',
    #                               '3x3_MBConv3',    '3x3_MBConv6',    '5x5_MBConv3',   '5x5_MBConv6',  '7x7_MBConv3',  '7x7_MBConv6',
    #                               '3x3_ResConv',    '5x5_ResConv',    '7x7_ResConv',
    #                               '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8',
    #                               '3x3_SpaConv',    '5x5_SpaConv',    '7x7_SpaConv']
    # elif args.candidates_type == 'Res_Group_Spa_MBConv':
    #     args.conv_candidates  = [ '3x3_MBConv3',    '3x3_MBConv6',    '5x5_MBConv3',   '5x5_MBConv6',  '7x7_MBConv3',  '7x7_MBConv6',
    #                               '3x3_ResConv',    '5x5_ResConv',    '7x7_ResConv',
    #                               '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8',
    #                               '3x3_SpaConv',    '5x5_SpaConv',    '7x7_SpaConv']
    if candidates_type == 'whole':
        conv_candidates = ['3x3_MBConv2', '3x3_MBConv4', '5x5_MBConv6',
                           '3x3_ResConv', '5x5_ResConv', '7x7_ResConv',
                           '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8',
                           '3x3_SpaConv', '5x5_SpaConv', '7x7_SpaConv']

    elif candidates_type == 'Res_Group':
        conv_candidates  = [ '3x3_ResConv',    '5x5_ResConv',    '7x7_ResConv',
                             '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8']
    elif candidates_type ==  'Res_Group_Spa':
        conv_candidates  = [ '3x3_ResConv',    '5x5_ResConv',    '7x7_ResConv',
                             '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8',
                             '3x3_SpaConv',    '5x5_SpaConv',    '7x7_SpaConv']
    elif candidates_type == 'Res_Group_Spa_MBConv':
        conv_candidates  = [ '3x3_MBConv2',    '3x3_MBConv4',    '5x5_MBConv6',
                             '3x3_ResConv',    '5x5_ResConv',    '7x7_ResConv',
                             '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8',
                             '3x3_SpaConv',    '5x5_SpaConv',    '7x7_SpaConv']
    elif candidates_type == 'ResConv':
        conv_candidates  = [ '3x3_ResConv', '5x5_ResConv', '7x7_ResConv']
    elif candidates_type == 'Proxyless':
        conv_candidates  = [ '3x3_MBConv3', '3x3_MBConv6',
                             '5x5_MBConv3', '5x5_MBConv6',
                             '7x7_MBConv3', '7x7_MBConv6', ]
    elif candidates_type == 'MBInverted':
        conv_candidates  = [ '3x3_MBConv2', '3x3_MBConv4', '3x3_MBConv6']
    elif candidates_type == 'GroupConv':
        conv_candidates  = [ '3x3_GroupConv2', '3x3_GroupConv4', '3x3_GroupConv8']
    elif candidates_type == 'SpaConv':
        conv_candidates  = [ '3x3_SpaConv', '5x5_SpaConv', '7x7_SpaConv']
    elif candidates_type == 'DepthSC':
        conv_candidates  = [ '3x3_DepthSC', '5x5_DepthSC', '7x7_DepthSC']
    elif candidates_type == 'single':
        conv_candidates  = [ '3x3_ResConv']
    elif candidates_type == 'single_Spa':
        conv_candidates  = [ '3x3_SpaConv']

    elif candidates_type == '3x3_DepthSC':
        conv_candidates  = [ '3x3_DepthSC']
    elif candidates_type == '5x5_DepthSC':
        conv_candidates  = [ '5x5_DepthSC']
    elif candidates_type == '7x7_DepthSC':
        conv_candidates  = [ '7x7_DepthSC']
    elif candidates_type == '3x3_MBConv2':
        conv_candidates  = [ '3x3_MBConv2']
    elif candidates_type == '3x3_MBConv4':
        conv_candidates  = [ '3x3_MBConv4']
    elif candidates_type == '3x3_MBConv6':
        conv_candidates  = [ '3x3_MBConv6']
    elif candidates_type == '3x3_ResConv':
        conv_candidates  = [ '3x3_ResConv']
    elif candidates_type == '5x5_ResConv':
        conv_candidates  = [ '5x5_ResConv']
    elif candidates_type == '7x7_ResConv':
        conv_candidates  = [ '7x7_ResConv']
    elif candidates_type == '3x3_GroupConv2':
        conv_candidates  = [ '3x3_GroupConv2']
    elif candidates_type == '3x3_GroupConv4':
        conv_candidates  = [ '3x3_GroupConv4']
    elif candidates_type == '3x3_GroupConv8':
        conv_candidates  = [ '3x3_GroupConv8']
    elif candidates_type == '3x3_SpaConv':
        conv_candidates  = [ '3x3_SpaConv']
    elif candidates_type == '5x5_SpaConv':
        conv_candidates  = [ '5x5_SpaConv']
    elif candidates_type == '7x7_SpaConv':
        conv_candidates  = [ '7x7_SpaConv']

    return  conv_candidates

def iter_layer_compute(meta,  num):

        if num - meta < 0:
            iteration = 0
            layer = num
        elif num - (2 * meta - 1) < 0:
            iteration = 1
            layer = num - meta
        elif num - (3 * meta - 3) < 0:
            iteration = 2
            layer = num - (2 * meta - 1)
        elif num - (4 * meta - 6) < 0:
            iteration = 3
            layer = num - (3 * meta - 3)
        elif num - (5 * meta - 10) < 0:
            iteration = 4
            layer = num - (4 * meta - 6)
        elif num - (6 * meta - 15) < 0:
            iteration = 5
            layer = num - (5 * meta - 10)

        return iteration, layer



def middle_block_latency_decoder(x, net, i, Block_OP_total, key_name, conv_name, Final_conv_name, model_name=None):

    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    B_i, C_i, H_i, W_i = np.shape(x)
    time_list, x       = net.blocks[i].get_latency(x)
    B_o, C_o, H_o, W_o = np.shape(x)
    try:
        Block_OP_total[net.blocks[i].mobile_inverted_conv._get_name() + "_" + str(i)] = time_list
    except:
        Block_OP_total[net.blocks[i]._get_name() + "_" + str(i)] = time_list

    if   model_name == 'Super_all':
        for m in range(len(key_name)):
                Final_conv_name[key_name[m].split('_')[1][:-1] + str(i) + str(m)] = \
                                                                                key_name[m].split('_')[1][:-1]  + '-input:' + \
                                                                                str(H_i)   + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                                                                                str(H_o)   + 'x' + str(W_o) + 'x' + str(C_o) + \
                                                                                "-expand:" + str(conv_name[key_name[m]].split('expand:')[1].split(',')[0]) + \
                                                                                '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                                                                                '-stride:' + str(1) + \
                                                                                '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0])  + ',' \
                                                                                  'value:' + str(time_list[m])

    elif model_name == 'Super_half':
        for m in range(len(key_name)):
                if i == len(net.blocks)-1:
                    Final_conv_name['block_end'] = \
                                                                                net.blocks[i].candidate_ops[0]._get_name() + '-input:' + \
                                                                                str(H_i)   + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                                                                                str(H_o)   + 'x' + str(W_o) + 'x' + str(C_o) + \
                                                                                "-expand:" + str(conv_name[key_name[m]].split('expand:')[1].split(',')[0]) + \
                                                                                '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                                                                                '-stride:' + str(1) + \
                                                                                '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0]) + ',' \
                                                                                                                                                           'value:' + str(
                                                                                    time_list[m])


                    break
                else:
                    Final_conv_name[key_name[m].split('_')[1][:-1] + str(i) + str(m)] = \
                                                                                key_name[m].split('_')[1][:-1]  + '-input:' + \
                                                                                str(H_i)   + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                                                                                str(H_o)   + 'x' + str(W_o) + 'x' + str(C_o) + \
                                                                                "-expand:" + str(conv_name[key_name[m]].split('expand:')[1].split(',')[0]) + \
                                                                                '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                                                                                '-stride:' + str(1) + \
                                                                                '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0])  + ',' \
                                                                                  'value:' + str(time_list[m])

    return  x

def middle_block_latency_decoder_half(x, net, i, Block_OP_total, key_name, conv_name, Final_conv_name):

    x                  = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    B_i, C_i, H_i, W_i = np.shape(x)
    time_list, x       = net.blocks[len(net.blocks)-2-i].get_latency(x)
    B_o, C_o, H_o, W_o = np.shape(x)
    try:
        Block_OP_total[net.blocks[len(net.blocks)-2-i].mobile_inverted_conv._get_name() + "_" + str(len(net.blocks)+i)] = time_list
    except:
        Block_OP_total[net.blocks[len(net.blocks)-2-i]._get_name() + "_" + str(len(net.blocks)+i)] = time_list
    for m in range(len(key_name)):
        Final_conv_name[key_name[m].split('_')[1][:-1] + str(len(net.blocks)+i) + str(m) if i!=(len(net.blocks) -2) else 'block_end'] = \
                                                                            key_name[m].split('_')[1][:-1] if i!=(len(net.blocks) -2) else (net.blocks[0].candidate_ops[0]._get_name()) + '-input:' + \
                                                                            str(H_i) + 'x' + str(W_i) + 'x' + str(C_i) + '-output:' + \
                                                                            str(H_o) + 'x' + str(W_o) + 'x' + str(C_o) + \
                                                                            "-expand:" + str(conv_name[key_name[m]].split('expand:')[1].split(',')[0]) + \
                                                                            '-kernel:' + str(conv_name[key_name[m]].split('kernel:')[1].split(',')[0]) + \
                                                                            '-stride:' + str(1) + \
                                                                            '-group:'  + str(conv_name[key_name[m]].split('group:')[1].split(',')[0]) + ',' \
                                                                              'value:' + str(time_list[m])

        if len(net.blocks)-2-i == 0:  ### conv-0 只有一个Block
            break


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)


def save_path(gpu, dataset, model, candidates_type):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    save_dir = "%s_%s_%s_%s_%s" % (gpu, dataset, model,candidates_type, dt_string)
    return save_dir

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict       = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def save_train_log(args, save_dir):
    dict_args  = vars(args)
    args_key   = list(dict_args.keys())
    args_value = list(dict_args.values())
    with open(save_dir+'/'+'parameters.txt' ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def batch_intersection_union(output, target, nclass):

    mini    = 1
    maxi    = 1
    nbins   = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


def decode(file, tot_iteration):
    """
    decode Phase1 searched genes from txt file
    :tp: type of block, encoder/decoder
    :it: iteration time
    :level
    """
    encoder  = []
    subcoder =[]
    num = tot_iteration
    for  _ in range(tot_iteration):
        for i in range(num):
            subcoder.append([None])
        encoder.append(subcoder)
        subcoder = []
        num -= 1
    # encoder = [[None for _ in range(tot_iteration)] for _ in range(tot_iteration)]
    with open(file, 'r') as f:
        for idx, l in enumerate(f.readlines()):
            if idx == 0:
                continue
            s = l.strip().split('_')
            tp, it, level = s[:3]
            pre = [int(v) for v in s[3:]]
            if tp == 'enc':
                if it == '0':
                    pre = [0]
                encoder[int(it)][int(level)] = pre
    codes = []
    for i in range(tot_iteration):
        codes.append(encoder[i])
    print(codes)
    return codes



def decode_gene(gene_path):
    more_down = 'False'

    with open(gene_path ,'r') as  f:
        line = f.readline()
        while line:
            line = f.readline()
            if   'channel'       in line.split(':')[0]:
                channel       =  str(line.split('--')[1].split('\n')[0])
            elif 'backbone'      in line:
                backbone      = line.split('--')[1].split('\n')[0]
            elif 'iter'          in line:
                iterations    = int(line.split('--')[1])
            elif 'conv_type'     in line:
                conv_type     = line.split('--')[1].split('\n')[0]
            elif 'add_decoder'   in line:
                add_decoder   = line.split('--')[1].split('\n')[0]
            elif 'add_encoder0'  in line:
                add_encoder0  = line.split('--')[1].split('\n')[0]
            elif 'more_down'     in line:
                more_down     = line.split('--')[1].split('\n')[0]

    return backbone, channel,  iterations, conv_type, add_decoder, add_encoder0, more_down


def load_param(backbone):

    if   backbone == 'resnet_10':
        num_blocks  = [1, 1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks  = [2, 2, 2, 2, 1]
    elif backbone == 'resnet_34':
        num_blocks  = [3, 4, 6, 3, 1]

    return num_blocks

def BFS(iters, codes, add_decoder, add_encoder0):
    """
    check if a block is skipped or not
    BFS from the last extraction stage to the first one
    """
    new_codes  = []
    pre_iter   = [0] # last block at last extraction stage
    # pre_iter = [[ _  for _ in range(iters-m+1)]  for m in range(iters, 0, -1)]
    if add_decoder=='True':
        for i in range(iters - 1, 0, -1):
            if [iters+1-i][0] not in codes[i][-1]:
                codes[i][-1].append(iters+1-i)


    for i in range(iters-1, 0, -1): # for each extraction stage pair
        # pre_iter = [m for m in range(iters-i)]

        temp      = {}
        temp_code = []
        for level in pre_iter:          # for each non-skipped block
            for out in codes[i][level]: # for each outgoing skip
                if out == 0:  # if the skip is sequential
                    if level == 0:
                        continue
                    else:
                        pre_iter.append(level-1)
                else: # if not sequential
                    temp[out - 1] = 1

        for k in temp.keys():
            temp_code.append(k)

        temp_code = list(set(temp_code))
        pre_iter  = temp_code[:]    # to be searched in next round
        new_codes.append(pre_iter)  # append non-skipped blocks at preceding stage to new_codes

    new_codes     = [[0]] + new_codes #[:4] + [[0, 1, 2, 3]]
    new_codes     = new_codes[::-1]
    # skip_codes    = [[True for _ in range(iters)] for _ in range(iters)]
    skip_codes    = [[True for _ in range(iters-m)] for m in range(iters)]


    if add_encoder0:
        ### 现在的encoder0，全部为false
        for i in range(iters):
            if i == 0:
                for code in range(iters):
                    skip_codes[i][code] = False
            else:
                new_codes[i] = list(set(new_codes[i]))
                for code in new_codes[i]:
                    skip_codes[i][code] = False
        print('These blocks as skipped:', skip_codes)
    else:
        ## 之前的encoder0，全部由上一个block决定true还是false
        for i in range(iters):
            new_codes[i] = list(set(new_codes[i]))
            for code in new_codes[i]:
                skip_codes[i][code] = False
        print('These blocks as skipped:', skip_codes)
    return skip_codes



class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0



class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos


class PD_FA():
    def __init__(self, nclass, bins, crop_size):
        super(PD_FA, self).__init__()
        self.nclass    = nclass
        self.bins      = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA        = np.zeros(self.bins+1)
        self.PD        = np.zeros(self.bins + 1)
        self.target    = np.zeros(self.bins + 1)
        self.crop_size = crop_size
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (preds.size()[2], preds.size()[3]))
            labelss  = np.array((labels).cpu()).astype('int64') # P
            # labelss  = np.reshape (labelss , (self.crop_size,self.crop_size))
            labelss  = np.reshape (labelss , (preds.size()[2], preds.size()[3]))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num,crop_size):

        Final_FA =  self.FA / ((crop_size * crop_size) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])


def load_dataset_eva (root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids,test_txt


def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)

def save_Ori_intensity_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix, size_2, size_3):

    predsss_real = np.array((pred > 0).cpu()).astype('int64')*np.array((pred).cpu()).astype('int64')
    predsss_real = np.uint8(predsss_real)[0][0]
    predsss_255  = np.uint8(np.array((pred > 0).cpu()).astype('int64')*255)


    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())[0][0]

    # img = Image.fromarray(predsss_real.reshape(size_2, size_3))
    # img.save(target_image_path + '/' + '%s_Pred_real' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(predsss_255.reshape(size_2, size_3))
    img.save(target_image_path + '/' + '%s_Pred_255' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(predsss_real.reshape(size_2, size_3))
    img.save(target_image_path + '/' + '%s_Pred_WOT' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(size_2, size_3))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

def SoftIoULoss(pred, target):
    pred         = torch.sigmoid(pred)
    smooth       = 1
    intersection = pred * target
    loss         = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
    loss         = 1 - loss.mean()
    return loss


def load_dataset (root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids,test_txt


class TrainSetLoader(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1
    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png',WS='Full',n_segments=10, compactness=20):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.transform   = transform
        self._items      = img_id
        self.masks       = dataset_dir+'/'+'masks'
        self.images      = dataset_dir+'/'+'images'
        self.base_size   = base_size
        self.crop_size   = crop_size
        self.suffix      = suffix
        self.WS          = WS
        self.n_segments  = n_segments
        self.compactness = compactness

    def _sync_transform(self, img, mask, img_id):
        # random mirror
        if random.random() < 0.5:
            img   = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img  = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img  = ImageOps.expand(img,  border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1   = random.randint(0, w - crop_size)
        y1   = random.randint(0, h - crop_size)
        img  = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))


        img, mask = np.array(img), np.array(mask, dtype=np.float32)


        return img, mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]                      # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img  = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)


        # synchronized transform
        img, mask = self._sync_transform(img, mask, img_id)

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)
        # plt.show()

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0



        return img, torch.from_numpy(mask)  #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)


        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)


class InferenceLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(InferenceLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        # img  = img.resize ((base_size, base_size), Image.BILINEAR)
        # mask = mask.resize((base_size, base_size), Image.NEAREST)


        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id     = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2


        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)
        ori_img   = img.copy()

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        # print(img_path)
        # print('np.shape(ori_img)：',np.shape(ori_img))
        # print('np.shape(mask)：',   np.shape(mask))

        return img, torch.from_numpy(mask), ori_img, img_id # img_id[-1]

    def __len__(self):
        return len(self._items)

class VisualizationLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(VisualizationLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img):
        base_size = self.base_size
        # final transform
        img = np.array(img)
        return img

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id     = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2

        img  = Image.open(img_id).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = self._testval_sync_transform(img)
        ori_img   = img.copy()

        if self.transform is not None:
            img = self.transform(img)


        return img, ori_img, img_id # img_id[-1]

    def __len__(self):
        return len(self._items)

class InferenceLoader_resize(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(InferenceLoader_resize, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)


        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id     = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2


        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)
        ori_img   = img.copy()

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        # print(img_path)
        # print('np.shape(ori_img)：',np.shape(ori_img))
        # print('np.shape(mask)：',   np.shape(mask))

        return img, torch.from_numpy(mask), ori_img, img_id # img_id[-1]

    def __len__(self):
        return len(self._items)