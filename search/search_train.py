from  search.models      import ImagenetRunConfig, Cifar10RunConfig, UcmRunConfig, SIRSTRunConfig
from   search.nas_manager import *
from   search.models.super_nets.super_proxyless_SIRST import SuperProxylessNASNets
import os
from search.utils.utils import save_path,conv_name_define,save_train_log, weights_init_xavier
from search.utils.utils import  operation_choose, random_operation_choose

def search_func(args):
    from torchstat import stat

    # ref values
    ref_values = {
        'flops': {
            '16.00': 59  * 1e6,
            '32.00': 97  * 1e6,
            '48.00': 209 * 1e6,
            'two':   0.9 * 1e9},

        # ms
        'mobile': {
            '16.00': 80,
            '32.00': 160},

        'cpu': {'16.00':  10,
                '32.00':  20,
                'two'  :  90},


        'gpu': {'16.00': 5,
                '32.00': 10,
                'two'  : 4},

        'edge-cpu': {'16.00': 10,
                    '32.00': 20,
                    'two': 1200},

        'edge-gpu': {'16.00': 5,
                    '32.00': 10,
                    'two': 100},

        'params': {
            '16.00': 1   * 1e6,
            '32.00': 2   * 1e6,
            '48.00': 3   * 1e6,
            'two':   2.5 * 1e6},
    }

    save_dir  = save_path(args.gpu, args.dataset, args.model_name, args.candidates_type)
    args.path = args.path + '/' + save_dir

    os.makedirs(args.path, exist_ok=True)
    save_train_log(args, args.path)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov, }

    run_config = SIRSTRunConfig(
        **args.__dict__)

    if args.random_choose == 'True':
        args.conv_candidates = random_operation_choose(args.candidates_type, args.iterations,args.random_choose)
    else:
        args.conv_candidates = operation_choose(args.candidates_type)

    args.conv_name          = conv_name_define(args.conv_candidates, args.iterations)


    super_net               = SuperProxylessNASNets(
        conv_type       = args.conv_type,         iterations=args.iterations,  backbone   =args.backbone,     in_channel = args.in_channel,
        conv_candidates = args.conv_candidates,   n_classes =args.num_class,   channel_num=args.channel_num,
        model_name      = args.model_name,        gene      =args.gene,        add_decoder=args.add_decoder,  add_encoder0=args.add_encoder0,  more_down = args.more_down )
    # stat = stat(super_net.cpu(), (3, 256, 256))

    # build arch search config from args
    if args.arch_opt_type  == 'adam':
        args.arch_opt_param = {'betas': (args.arch_adam_beta1, args.arch_adam_beta2), 'eps': args.arch_adam_eps, }
    else:
        args.arch_opt_param = None

    if args.target_hardware is None:
        args.ref_value = None
    else:
        # args.ref_value = ref_values[args.target_hardware]['%.2f' % args.channel_num]
        if args.grad_reg_loss_type != 'add#linear_abs':
            args.ref_value = ref_values[args.target_hardware][args.channel_num]
    print('args.ref_value: ',args.ref_value)



    if args.arch_algo == 'grad':
        from nas_manager import GradientArchSearchConfig
        if  args.grad_reg_loss_type   == 'add#linear':
            args.grad_reg_loss_params  = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type  == 'mul#log':
             args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta, }
        else:
            args.grad_reg_loss_params = None
        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    else:
        raise NotImplementedError

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))

    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path,
                                                   super_net,
                                                   args.crop_size,
                                                   run_config,
                                                   arch_search_config,
                                                   args.conv_name,
                                                   args.target_hardware,
                                                   args.fast,
                                                   args.model_name,
                                                   args.iterations,
                                                   args.mode,
                                                   args.gene,
                                                   args.candidates_type,
                                                   args.random_choose,)
    # warmup
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs, crop_size=args.crop_size, lr=args.init_lr)

    # # joint training
    arch_search_run_manager.train()
    return args.path


def retrain_func(args, search_log):
    from torchstat import stat

    pretrain_log = search_log
    args.path    = pretrain_log + '_Retrain'
    for i in range(100):
        if i == 0:
            if os.path.exists(args.path):
                args.path = args.path + "_" + str(2)
                continue
        else:
            if os.path.exists(args.path):
                args.path = args.path.split('_Retrain_')[0] + '_Retrain_' + str(i+2)
                continue

    os.makedirs(args.path, exist_ok=True)

    # prepare run config
    run_config_path = '%s/learned_net/run.config' % pretrain_log
    if os.path.isfile(run_config_path):
        # load run config from file
        run_config  = json.load(open(run_config_path, 'r'))
        run_config  = SIRSTRunConfig(**run_config)
        run_config.dataset          = args.dataset
        run_config.split_method     = args.split_method
        run_config.lr_schedule_type = args.retrain_lr_schedule_type
        run_config.n_epochs         = args.retrain_epoch
        run_config.root             = args.root

        if args.retrain_valid_size:
            run_config.valid_size = args.retrain_valid_size


    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path  = '%s/learned_net/net.config' % pretrain_log
    if os.path.isfile(net_config_path):
        # load net from file
        from models import get_net_by_name
        net_config   = json.load(open(net_config_path, 'r'))
        net          = get_net_by_name(net_config['name']).build_from_config(net_config, args.gene)
        net.apply(weights_init_xavier)
        # stat         = stat(net, (3, 256, 256))

    # if isinstance(net, nn.DataParallel):  ## stat的输出是对的，这个不对
    #     total_params = count_parameters(net.module)
    # else:
    #     total_params = count_parameters(net)
    # print('Total training params: %.2fM' % (total_params / 1e6))


    # build run manager
    run_manager      = RunManager(args.path, net, run_config, second_train=True)
    run_manager.save_config(print_info=True)
    flops,total_params  = run_manager.net_flops()
    print('flops:', flops/10**9, 'G')
    print('total_params:', total_params/10**6, 'M')

    # train
    print('Start training')
    run_manager.train(fixed_lr=args.retrain_fixed_lr)
    run_manager.save_model()

    output_dict = {}

    # validate
    if run_config.valid_size:
        print('Test on validation set')
        loss, validate_IoU = run_manager.validate(is_test=False)
        log = 'valid_loss: %f\t validate_IoU: %f' % (loss, validate_IoU)
        run_manager.write_log(log, prefix='valid')
        output_dict = {
            **output_dict,
            'valid_loss': ' % f' % loss, 'validate_IoU': ' % f' % validate_IoU,
            'valid_size': run_config.valid_size
        }

    # test
    print('Test on test set')
    loss, validate_IoU = run_manager.validate(is_test=True)
    log = 'test_loss: %f\t validate_IoU: %f' % (loss, validate_IoU)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'validate_IoU': '%f' % validate_IoU,
    }
    json.dump(output_dict, open('%s/output' % args.path, 'w'), indent=4)


    return args.path
def inference(args, search_log, inference_path):
    from torchstat import stat
    pretrain_log = search_log
    args.path    = inference_path

    # prepare run config
    run_config_path = '%s/learned_net/run.config' % pretrain_log
    if os.path.isfile(run_config_path):
        # load run config from file
        run_config  = json.load(open(run_config_path, 'r'))
        run_config  = SIRSTRunConfig(**run_config)
        run_config.lr_schedule_type = args.retrain_lr_schedule_type
        run_config.n_epochs         = args.retrain_epoch
        run_config.root             = args.root
        run_config.dataset          = args.dataset
        run_config.split_method     = args.split_method
        if args.retrain_valid_size:
            run_config.valid_size = args.retrain_valid_size



    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path  = '%s/learned_net/net.config' % pretrain_log
    if os.path.isfile(net_config_path):
        # load net from file
        from search.models import get_net_by_name
        net_config   = json.load(open(net_config_path, 'r'))
        net          = get_net_by_name(net_config['name']).build_from_config(net_config, args.gene)
        checkpoint   = torch.load(inference_path + '/checkpoint' + '/model_best.pth.tar')
        net.load_state_dict(checkpoint['state_dict'])
        # stat         = stat(net, (3, 256, 256))
    run_manager      = RunManager(args.path, net, run_config, second_train=True)
    run_manager.inference(args, is_test=False)

    # gpu_avg_time, cpu_avg_time = run_manager.inference_latency(args, is_test=False)

    # flops, total_params  = run_manager.net_flops()
    # print('flops:', flops/10**9, 'G')
    # print('total_params:', total_params/10**6, 'M')
    # print('gpu_avg_time：',gpu_avg_time)
    # print('cpu_avg_time：',cpu_avg_time)


def visualization(args, search_log, inference_path, model_name):
    from torchstat import stat
    pretrain_log = search_log
    args.path    = inference_path

    # prepare run config
    run_config_path = '%s/learned_net/run.config' % pretrain_log
    if os.path.isfile(run_config_path):
        # load run config from file
        run_config  = json.load(open(run_config_path, 'r'))
        run_config  = SIRSTRunConfig(**run_config)
        run_config.lr_schedule_type = args.retrain_lr_schedule_type
        run_config.n_epochs         = args.retrain_epoch
        run_config.root             = args.root
        run_config.dataset          = args.dataset
        run_config.split_method     = args.split_method
        if args.retrain_valid_size:
            run_config.valid_size = args.retrain_valid_size


    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path  = '%s/learned_net/net.config' % pretrain_log
    if os.path.isfile(net_config_path):
        # load net from file
        from models import get_net_by_name
        net_config   = json.load(open(net_config_path, 'r'))
        net          = get_net_by_name(net_config['name']).build_from_config(net_config, args.gene)
        checkpoint   = torch.load(inference_path + '/checkpoint' + '/model_best.pth.tar')
        net.load_state_dict(checkpoint['state_dict'])
        # stat         = stat(net, (3, 256, 256))
    run_manager      = RunManager(args.path, net, run_config, second_train=True)
    run_manager.visualization(args, is_test=False, model_name=model_name)

