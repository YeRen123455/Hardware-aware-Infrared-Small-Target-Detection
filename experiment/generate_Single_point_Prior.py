import os
from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import  Image
import random
from skimage import measure
import cv2

def generate_random_bb_single_point(img, mask, img_name, range_min, range_max, save_dir):
    print('img_name:',img_name)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    label = measure.label(mask, connectivity=2)
    coord_mask = measure.regionprops(label)
    Ymin_f = []
    Ymax_f = []
    Xmin_f = []
    Xmax_f = []
    centroid_label_y = []
    centroid_label_x = []
    target_area = 0

    for target_num in range(len(coord_mask)):
        centroid_label = np.array(list(coord_mask[target_num].centroid))

        Height_min = random.randint(range_min, range_max)
        Height_max = random.randint(range_min, range_max)
        Width_min  = random.randint(range_min, range_max)
        Width_max  = random.randint(range_min, range_max)

        Ymin_f.append(int(max((centroid_label[0] - Height_min),0)))
        Ymax_f.append(int(min((centroid_label[0] + Height_max), img.shape[0]-1)))
        Xmin_f.append(int(max((centroid_label[1] - Width_min), 0)))
        Xmax_f.append(int(min((centroid_label[1] + Width_max),  img.shape[1]-1)))
        centroid_label_y.append(int(centroid_label[0]))
        centroid_label_x.append(int(centroid_label[1]))

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(img[Ymin_f[target_num]:Ymax_f[target_num], Xmin_f[target_num]:Xmax_f[target_num]])
        # plt.subplot(122)
        # plt.imshow(mask[Ymin_f[target_num]:Ymax_f[target_num], Xmin_f[target_num]:Xmax_f[target_num]])
        # plt.show()


    save_content = {'name': img_name,
                    'Ymin_f': Ymin_f, 'Ymax_f': Ymax_f, 'Xmin_f': Xmin_f, 'Xmax_f': Xmax_f,
                    'centroid_label_y': centroid_label_y,  'centroid_label_x': centroid_label_x,}
    print(save_dir + img_name+'.npy')
    np.save(save_dir + img_name+'.npy', save_content)


def generate_random_size_prior(img, mask, img_name, range_min, range_max, save_dir):
    print('img_name:',img_name)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if img_name == 'Misc_83':
        print()
    label = measure.label(mask, connectivity=2)
    coord_mask = measure.regionprops(label)
    Ymin_f = []
    Ymax_f = []
    Xmin_f = []
    Xmax_f = []
    centroid_label_y = []
    centroid_label_x = []
    target_type      = []
    target_area = 0
    if len(coord_mask)!=0:  ## mask usually = 0 on NUDT-SIRST-sea dataset
        for target_num in range(len(coord_mask)):
            centroid_label = np.array(list(coord_mask[target_num].centroid))
            single_area = coord_mask[target_num].area
            if single_area <= 9:
                Height_min     = random.randint(3, 5)
                Height_max     = random.randint(3, 5)
                Width_min      = random.randint(3, 5)
                Width_max      = random.randint(3, 3)
                target_type.append('Point')
            elif 9 < single_area <= 81:

                Height_min = random.randint(10, 11)
                Height_max = random.randint(10, 11)
                Width_min  = random.randint(10, 11)
                Width_max  = random.randint(10, 11)

                # Height_min     = random.randint(10, 15)
                # Height_max     = random.randint(10, 15)
                # Width_min      = random.randint(10, 15)
                # Width_max      = random.randint(10, 15)

                # Height_min     = random.randint(10, 15)
                # Height_max     = random.randint(10, 15)
                # Width_min      = random.randint(10, 15)
                # Width_max      = random.randint(10, 15)

                # Height_min     = random.randint(15, 20)
                # Height_max     = random.randint(15, 20)
                # Width_min      = random.randint(15, 20)
                # Width_max      = random.randint(15, 20)
                target_type.append('Spot')

            elif 81 < single_area:
                # Height_min     = 15
                # Height_max     = 15
                # Width_min      = 15
                # Width_max      = 15

                Height_min     = random.randint(15, 25)
                Height_max     = random.randint(15, 25)
                Width_min      = random.randint(15, 25)
                Width_max      = random.randint(15, 25)

                # Height_min     = random.randint(20, 25)
                # Height_max     = random.randint(20, 25)
                # Width_min      = random.randint(20, 25)
                # Width_max      = random.randint(20, 25)
                target_type.append('Extended')

            Ymin_f.append(int(max((centroid_label[0] - Height_min),0)))
            Ymax_f.append(int(min((centroid_label[0] + Height_max), img.shape[0]-1)))
            Xmin_f.append(int(max((centroid_label[1] - Width_min), 0)))
            Xmax_f.append(int(min((centroid_label[1] + Width_max),  img.shape[1]-1)))
            centroid_label_y.append(int(centroid_label[0]))
            centroid_label_x.append(int(centroid_label[1]))
            target_area += single_area

            print('single_area:', single_area)

        save_content = {'name': img_name,
                        'Ymin_f': Ymin_f, 'Ymax_f': Ymax_f, 'Xmin_f': Xmin_f, 'Xmax_f': Xmax_f,
                        'centroid_label_y': centroid_label_y,  'centroid_label_x': centroid_label_x,
                        'target_type':target_type}
        print(save_dir + img_name+'.npy')
        np.save(save_dir + img_name+'.npy', save_content)

    elif len(coord_mask) == 0:  ## mask usually = 0 on NUDT-SIRST-sea dataset
        Ymin_f = 0
        Ymax_f = 0
        Xmin_f = 0
        Xmax_f = 0
        centroid_label_y = 0
        centroid_label_x = 0
        target_type  = 'non_target'

        save_content = {'name': img_name,
                        'Ymin_f': Ymin_f, 'Ymax_f': Ymax_f, 'Xmin_f': Xmin_f, 'Xmax_f': Xmax_f,
                        'centroid_label_y': centroid_label_y, 'centroid_label_x': centroid_label_x,
                        'target_type': target_type}
        print(save_dir + img_name + '.npy')
        np.save(save_dir + img_name + '.npy', save_content)


if __name__ == "__main__":

    dataset_root      = '/media/gfkd/sda/Dataset'
    dataset           = 'NUAA-SIRST'

    mask_root         = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs'
    event_dir         = '0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_01_11_2023_11_34_58_Retrain_2'
    original_img_dir  = mask_root    + '/' + event_dir  + '/' + 'visulization_result'
    original_mask_dir = mask_root    + '/' + event_dir  + '/' + 'visulization_result'
    txt_dir           = dataset_root + '/' + dataset    + '/' + 'mode/' + 'test.txt'


    #######CRFV1 CRFV2
    save_dir  =  mask_root + '/' +  event_dir + '/' + 'image_info/'
    img_id    = []
    range_min = 0
    range_max = 0


    with open(txt_dir, "r") as f:
        line = f.readline()
        while line:

            img_id.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    # print(img_id)

    for img_num in range(len(img_id)):
        # print('img_id[img_num]:',img_id[img_num])
        img  = np.array(Image.open(original_img_dir  + '/' + img_id[img_num] + '.png').convert('RGB'))
        mask = np.array(Image.open(original_mask_dir + '/' + img_id[img_num] + '_Pred_255' + '.png').convert('RGB'))/255
        # mask = np.array(Image.open(original_mask_dir + '/' + img_id[img_num] + '_GT' + '.png').convert('RGB'))/255

        img_name = img_id[img_num]
        # generate_random_bb_single_point(img, mask, img_name, range_min, range_max, save_dir)
        generate_random_size_prior(img, mask, img_name, range_min, range_max, save_dir)

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()



