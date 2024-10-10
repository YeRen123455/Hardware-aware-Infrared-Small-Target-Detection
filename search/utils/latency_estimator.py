
import yaml
import os
import sys
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def download_url(url, model_dir='~/.torch/proxyless_nas', overwrite=False):
    target_dir  = url.split('//')[-1]
    target_dir  = os.path.dirname(target_dir)
    model_dir   = os.path.expanduser(model_dir)
    model_dir   = os.path.join(model_dir, target_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename    = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


class LatencyEstimator(object):
    def __init__(self, path=None,  initial_metric=None, metric_txt=None):
        # fname = download_url(url, overwrite=True)
        # fname = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST/search/config/mobile_trim.yaml'
        # if   target_hardware == 'cpu':
        #     fname = path + '/' + 'Latency_' + 'cpu.yaml'
        # elif target_hardware == 'gpu':
        #     fname = path + '/' + 'Latency_' + 'gpu.yaml'
        # with open(fname, 'r') as fp:
        #     self.lut = yaml.load(fp,Loader=yaml.FullLoader)

        self.initial_latency = initial_metric
        self.metric_txt      = metric_txt
        self.conv_name = []
        self.op_value  = []
        for key, item in self.metric_txt.items():
            if   'ResCon'    in key:
                self.conv_name.append(key+'_'+'kernel:'+item.split('kernel:')[1].split('-')[0])
            elif 'GroupConv' in key:
                self.conv_name.append(key+'_'+'group:'+item.split('group:')[1].split(',')[0])
            elif 'SpaCon' in key:
                self.conv_name.append(key+'_'+'kernel:'+item.split('kernel:')[1].split('-')[0])
            elif 'MBConv' in key:
                self.conv_name.append(key+'_'+'expand:'+item.split('expand:')[1].split('-')[0])

            self.op_value.append(float(item.split('value:')[1]))
        print()


    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, conv_name_latency=None, id=0, conv_num=None):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `Conv`  : The initial 3x3 conv with stride 2.
                2. `Conv_1`: The upsample 1x1 conv that increases num_filters by 4 times.
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param _input: input shape  (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        """
        try:
            latency   = float(self.initial_latency[conv_name_latency]) if conv_name_latency=='ConvLayer_Post_conv' else float(self.initial_latency[conv_name_latency][id])
            # if 'GroupConv' in  self.conv_name[conv_num] and 'group:8' in  self.conv_name[conv_num]:
            #     latency = latency*10
            #     # print(conv_num)

        except:
            latency = 0
            print(conv_name_latency, '------->', 'this block is skipped or latency_error')

        return latency


if __name__ == '__main__':
    est = LatencyEstimator()
    s = est.predict('expanded_conv', _input=(112, 112, 16), output=(56, 56, 24), expand=3, kernel=5, stride=2, idskip=0)
    print(s)
