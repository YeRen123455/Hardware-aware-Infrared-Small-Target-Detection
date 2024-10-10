
from search.models.normal_nets.proxyless_nets import ProxylessNASNets
from search.run_manager import RunConfig


def get_net_by_name(name):
    if   name == ProxylessNASNets.__name__:
        return ProxylessNASNets

    else:
        raise ValueError('unrecognized type of network: %s' % name)


class ImagenetRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='normal', **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'resize_scale': self.resize_scale,
            'distort_color': self.distort_color,
        }

class Cifar10RunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='cifar10', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='normal', **kwargs):
        super(Cifar10RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'resize_scale': self.resize_scale,
            'distort_color': self.distort_color,
        }

class UcmRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='cifar10', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='normal',mode=None, root=None, split_method=None,
                 base_size=None, crop_size=None, suffix=None, **kwargs):
        super(UcmRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency, mode, root, split_method, base_size,
            crop_size, suffix
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'resize_scale': self.resize_scale,
            'distort_color': self.distort_color,
        }

class SIRSTRunConfig(RunConfig):

    def __init__(self,                  n_epochs        =150,   init_lr             =0.05,    lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset   ='cifar10',  train_batch_size=256,   test_batch_size     =500,     valid_size      =None,
                 opt_type  ='sgd',      opt_param       =None,  weight_decay        =4e-5,    label_smoothing =0.1,      no_decay_keys='bn',
                 model_init='he_fout',  init_div_groups =False, validation_frequency=1,       print_frequency =10,
                 n_worker  =32,         resize_scale    =0.08,  distort_color       ='normal',id_mode         ='TXT',     root=None, split_method=None,
                 base_size =None,       crop_size       =None,  suffix              =None,    eval_batch_size =None,     **kwargs):
        super(SIRSTRunConfig, self).__init__(
            n_epochs,   init_lr,           lr_schedule_type,     lr_schedule_param,
            dataset,    train_batch_size,  test_batch_size,      valid_size,
            opt_type,   opt_param,         weight_decay,         label_smoothing,   no_decay_keys,
            model_init, init_div_groups,   validation_frequency, print_frequency,   id_mode,           root,   split_method,
            base_size,  crop_size, suffix, eval_batch_size
        )

        self.n_worker      = n_worker
        self.resize_scale  = resize_scale
        self.distort_color = distort_color

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size':  self.test_batch_size,
            'valid_size':       self.valid_size,
            'n_worker':         self.n_worker,
            'resize_scale':     self.resize_scale,
            'id_mode':          self.id_mode,
            'root':             self.root,
            'split_method':     self.split_method,
            'base_size':        self.base_size,
            'crop_size':        self.crop_size,
            'suffix':           self.suffix,
            'dataset':          self.dataset,
            'eval_batch_size':  self.eval_batch_size

        }