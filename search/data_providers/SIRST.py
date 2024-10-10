

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from   search.data_providers.base_provider import *
from   search.utils.utils import  *
from   torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class SIRST_DataProvider(Dataset):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None, id_mode='TXT', root=None, split_method=None, base_size=None,
                 crop_size=None, suffix=None, dataset=None, eval_batch_size=None):

        self.crop_size = crop_size

        if id_mode == 'TXT':
            dataset_dir = root + '/' + dataset
            self.train_img_ids, self.val_img_ids, test_txt = load_dataset(root, dataset, split_method)
        latency_list = [self.val_img_ids[0]]
        visualization_list = ['/media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/infrared_visualization/images/000001.png',
                              '/media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/infrared_visualization/images/000002.png',
                              '/media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/infrared_visualization/images/000003.png',
                              '/media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/infrared_visualization/images/000004.png',
                              '/media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/infrared_visualization/images/000005.png',
                              '/media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/infrared_visualization/images/000006.png',
                              '/media/lby/sda/NAS/proxylessnas-master-SIRST-new-final_share/infrared_visualization/images/000007.png']
        # Preprocess and load data
        if dataset == 'NUAA-SIRST-Old':
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        if dataset == 'NUAA-SIRST':
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.439, .439, .439], [.217, .217, .217])])
        elif dataset == 'NUDT-SIRST':
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.423, .423, .423],  [.217, .217, .217])])
        elif dataset == 'IRSTD-SIRST':
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( [.343, .343, .343],[.231, .231, .231])])

        trainset            = TrainSetLoader(dataset_dir,   img_id=self.train_img_ids, base_size=base_size, crop_size=crop_size, transform=input_transform, suffix=suffix)
        testset             = TestSetLoader(dataset_dir,    img_id=self.val_img_ids,   base_size=base_size, crop_size=crop_size, transform=input_transform, suffix=suffix)
        valset              = TestSetLoader(dataset_dir,    img_id=self.val_img_ids,   base_size=base_size, crop_size=crop_size, transform=input_transform, suffix=suffix)
        inferenceset        = InferenceLoader(dataset_dir,  img_id=self.val_img_ids,   base_size=base_size, crop_size=crop_size, transform=input_transform, suffix=suffix)
        inferenceset_resize = InferenceLoader_resize(dataset_dir,  img_id=self.val_img_ids,   base_size=base_size, crop_size=crop_size, transform=input_transform, suffix=suffix)
        inferenceset_latency= InferenceLoader(dataset_dir,  img_id=latency_list,   base_size=base_size, crop_size=crop_size, transform=input_transform, suffix=suffix)
        VisualziationLoader = VisualizationLoader(dataset_dir,  img_id=visualization_list,   base_size=base_size, crop_size=crop_size, transform=input_transform, suffix=suffix)

        self.train_data     = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True, num_workers=n_worker, drop_last=True)
        self.test_data      = DataLoader(dataset=testset,  batch_size=test_batch_size,                num_workers=n_worker, drop_last=False)
        self.val_data       = DataLoader(dataset=valset,   batch_size=eval_batch_size,                num_workers=n_worker, drop_last=False)
        self.inference_data = DataLoader(dataset=inferenceset,         batch_size=1, num_workers=n_worker, drop_last=False)
        self.inference_data_resize  = DataLoader(dataset=inferenceset_resize,  batch_size=1, num_workers=n_worker, drop_last=False)
        self.inference_data_latency = DataLoader(dataset=inferenceset_latency,  batch_size=1, num_workers=n_worker, drop_last=False)
        self.VisualziationLoader = DataLoader(dataset=VisualziationLoader,  batch_size=1, num_workers=n_worker, drop_last=False)

    @property
    def data_shape(self):
        return 3, self.crop_size, self.crop_size  # C, H, W
