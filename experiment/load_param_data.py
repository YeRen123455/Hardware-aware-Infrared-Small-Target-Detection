import numpy   as np
from   PIL     import Image
from   skimage import measure

def load_dataset (root, dataset, split_method, weak_dataset, size_prior):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            if size_prior=='Yes':
                    mask_info_dir      = '/media/gfkd/software/SIRST_Detection/WS_SIRST/dataset/final_dataset/' + dataset + '/image_info'
                    generated_mask_dir = '/media/gfkd/software/SIRST_Detection/WS_SIRST/dataset/final_dataset/' + dataset + '/evaluation_result/' + weak_dataset
                    generated_mask     = np.array(Image.open(generated_mask_dir + '/' + line.split('\n')[0]  + '.png' ))
                    generated_mask     = generated_mask / 255
                    label              = measure.label(generated_mask, connectivity=2)
                    coord_mask         = measure.regionprops(label)
                    target_area        = 0
                    mask_info          = np.load(mask_info_dir + '/' + line.split('\n')[0] + '.npy', allow_pickle=True)
                    target_info        = mask_info.item().get("target_type")

                    for target_num in range(len(coord_mask)):
                        single_area = coord_mask[target_num].area
                        target_area += single_area
                        # print('sing_target_area:', single_area)
                    target_area = target_area / len(coord_mask)
                    if   target_info =='Point':
                        if target_area >= 2:
                            train_img_ids.append(line.split('\n')[0])
                    elif target_info =='Spot':
                        if target_area >= 9:
                            train_img_ids.append(line.split('\n')[0])
                    elif target_info =='Extended':
                        if  target_area >= 81:
                            train_img_ids.append(line.split('\n')[0])
            else:
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


def load_param(channel_size, backbone, blocks_per_layer=4):
    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]
    elif channel_size == 'all_48':
        nb_filter = [48, 48, 48, 48, 48]
    elif channel_size == 'all_32':
        nb_filter = [32, 32, 32, 32, 32]
    elif channel_size == 'all_16':
        nb_filter = [16, 16, 16, 16, 16]

    if   backbone == 'resnet_10':
        num_blocks = [1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks = [2, 2, 2, 2]
    elif backbone == 'resnet_34':
        num_blocks = [3, 4, 6, 3]
    elif backbone == 'vgg_10':
        num_blocks = [1, 1, 1, 1]

    # ACM_channel = [8, 16, 32, 64]
    ACM_channel = [16, 32, 64, 128]

    # ACM_channel = [ 32, 64, 128, 256]

    ACM_layer_blocks = [blocks_per_layer] * 4

    return nb_filter, num_blocks,  ACM_channel, ACM_layer_blocks