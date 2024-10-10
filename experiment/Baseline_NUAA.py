from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import  os
import numpy as np
from PIL import  Image
import random
from skimage import measure
import cv2
from CRF_lib.DenseCRF import  dense_crf
from CRF_lib.imutils import  *
from PIL import Image, ImageOps, ImageFilter
from skimage import util, img_as_float, io    # 导入所需要的 skimage 库


def evaluation(txt_save_dir, save_dir, original_img_dir, Target_threshold, Repeated_num, Noise_type, Noise_intensity_amount, total_iter_num, SP_num):
    candidate_evaluation_type = ['with_type_wo_noise', 'wo_type_add_noise_crf', 'wo_type_add_noise_T']
    with open(txt_save_dir+ '/' + '%s_train_log_%d.txt' %  (save_dir, total_iter_num),   'a') as  f:
        f.write("Target_threshold:")
        f.write(str(Target_threshold))
        f.write("\t")
        f.write("Repeated_num:")
        f.write(str(Repeated_num))
        f.write("\t")
        f.write("Noise_type:")
        f.write(Noise_type)
        f.write("\t")
        f.write("Noise_intensity_amount:")
        f.write(str(Noise_intensity_amount))
        f.write("\t")
        f.write("SP_num:")
        f.write(str(SP_num))
        f.write("\t")
        f.write("\n")
    print("Target_threshold:  ",       str(Target_threshold),
          "Repeated_num:   ",          str(Repeated_num),
          'Noise_type:   ',            str(Noise_type),
          'Noise_intensity_amount:   ',str(Noise_intensity_amount),
          'SP_num:   ',                str(SP_num))


    for evaluation_num in range(len(candidate_evaluation_type)):
        with open(txt_save_dir + '/' + '%s_train_log_%d.txt' % (save_dir, total_iter_num),'a') as  f:
            f.write(candidate_evaluation_type[evaluation_num])
            f.write("\t""\t")

        original_mask_dir = dataset_root + '/' + dataset + '/' + 'masks'
        final_save_dir    = txt_save_dir + '/' + '%s'% save_dir
        # evaluation_mode = [ 'train_F', 'train_P', 'train_S', 'train_E']
        evaluation_mode = [ 'test']

        for item in range(len(evaluation_mode)):
            mode = evaluation_mode[item]
            if mode   == 'full':
                txt_dir = dataset_root + '/' + dataset + '/' + 'mode/' + 'full.txt'
            elif mode == 'train_F':
                txt_dir = dataset_root + '/' + dataset + '/' + 'mode/' + 'train.txt'
            elif mode == 'test':
                txt_dir = dataset_root + '/' + dataset + '/' + 'mode/' + 'test.txt'

            elif mode == 'train_P':
                txt_dir = dataset_root + '/' + dataset + '/' + 'mode/' + 'point_target_train.txt'
            elif mode == 'train_S':
                txt_dir = dataset_root + '/' + dataset + '/' + 'mode/' + 'spot_target_train.txt'
            elif mode == 'train_E':
                txt_dir = dataset_root + '/' + dataset + '/' + 'mode/' + 'extended_target_train.txt'

            test_img = []
            with open(txt_dir, "r") as f:
                line = f.readline()
                while line:
                    test_img.append(line.split('\n')[0])
                    line = f.readline()
                f.close()
            # print(test_img)
            mini = 1
            maxi = 1  # nclass
            nbins = 1  # nclass
            total_inter    = 0
            total_union    = 0
            for k in range(len(test_img)):
                WS_label   = Image.open(final_save_dir  + '/' + candidate_evaluation_type[evaluation_num] + '/' + test_img[k] + '_MCLC.png').convert('RGB')
                Full_label = Image.open(original_img_dir    + '/' + test_img[k] + '_GT.png').convert('RGB')
                # Full_label   = Image.open(final_save_dir  + '/' + candidate_evaluation_type[evaluation_num] + '/' + test_img[k] + '.png').convert('RGB')

                WS_label   = np.array(WS_label)
                Full_label = np.array(Full_label)

                if WS_label.ndim   == 3:
                    WS_label   = WS_label[:, :, 0]
                if Full_label.ndim == 3:
                    Full_label = Full_label[:, :, 0]

                WS_label     = (WS_label   > 0).astype('float32')
                Full_label   = (Full_label > 0).astype('float32')
                intersection = WS_label * ((WS_label == Full_label))  # TP

                # areas of intersection and union
                area_inter, _  = np.histogram(intersection, bins=nbins, range=(mini, maxi))
                area_pred,  _  = np.histogram(WS_label, bins=nbins, range=(mini, maxi))
                area_lab,   _  = np.histogram(Full_label, bins=nbins, range=(mini, maxi))
                area_union     = area_pred + area_lab - area_inter
                total_inter   += area_inter
                total_union   += area_union
            with open(txt_save_dir + '/' + '%s_train_log_%d.txt' % (save_dir, total_iter_num),   'a') as  f:
                f.write("mIoU:")
                f.write(mode)
                f.write(str(np.around(total_inter / total_union,6)))
                f.write("\t")
            print('candidate_evaluation_type: ', candidate_evaluation_type[evaluation_num])
            print('mode: ',mode, "mIoU: ", str(np.around(total_inter / total_union,6)))

        with open(txt_save_dir + '/' + '%s_train_log_%d.txt' % (save_dir, total_iter_num),'a') as  f:
            f.write("\n")
        print('-------------------------------')
    print('====================================================')


def super_pixel(img, mask, mask_info, segments_num):
    WS_mask    = np.zeros_like(img)[:,:,0]
    for target_num in range(len(mask_info.item().get("Ymax_f"))):
        Ymin_f           = mask_info.item().get("Ymin_f")[target_num]
        Ymax_f           = mask_info.item().get("Ymax_f")[target_num]
        Xmin_f           = mask_info.item().get("Xmin_f")[target_num]
        Xmax_f           = mask_info.item().get("Xmax_f")[target_num]

        centroid_label_x = mask_info.item().get("centroid_label_x")[target_num]
        centroid_label_y = mask_info.item().get("centroid_label_y")[target_num]

        centroid_coord_y = centroid_label_y - Ymin_f
        centroid_coord_x = centroid_label_x - Xmin_f


        crop_image = img [Ymin_f:Ymax_f, Xmin_f:Xmax_f]
        crop_mask  = mask[Ymin_f:Ymax_f, Xmin_f:Xmax_f]


        if   mask_info.item().get("target_type")[target_num] == 'Point':
            n_segments = segments_num[0]
            try:
                segments = slic(crop_image, n_segments=n_segments, compactness=20)
            except:
                print()
            # print('Point')
        elif mask_info.item().get("target_type")[target_num] == 'Spot':
            n_segments = segments_num[1]
            try:
                segments = slic(crop_image, n_segments=n_segments, compactness=20)
            except:
                print()
            # print('Spot')
        elif mask_info.item().get("target_type")[target_num] == 'Extended':
            n_segments = segments_num[2]
            segments = slic(crop_image, n_segments=n_segments, compactness=20)
            # print('Extended')
        try:
            out_boundary = mark_boundaries(crop_image, segments)
        except:
            print()
        centroid_coord_y = (crop_image.shape[0]-1  if crop_image.shape[0] <= centroid_coord_y else int(round(centroid_coord_y)))
        centroid_coord_x = (crop_image.shape[1]-1  if crop_image.shape[1] <= centroid_coord_x else int(round(centroid_coord_x)))

        area_num = segments[int(round(centroid_coord_y)), int(round(centroid_coord_x))]
        target_area  = np.where(segments == area_num)

        for point in range(np.array(target_area).shape[1]):
            WS_mask[Ymin_f + target_area[0][point], Xmin_f + target_area[1][point]] = 255
        crop_WS_mask = WS_mask[Ymin_f:Ymax_f, Xmin_f:Xmax_f]
    return  WS_mask, out_boundary, crop_image, crop_mask, crop_WS_mask, mask_info.item().get("target_type")


def super_pixel_noise_with_size_prior(img, mask, mask_info, segments_num, Noise_type, amount):
    WS_mask    = np.zeros_like(img)[:,:,0]
    for target_num in range(len(mask_info.item().get("Ymax_f"))):
        Ymin_f           = mask_info.item().get("Ymin_f")[target_num]
        Ymax_f           = mask_info.item().get("Ymax_f")[target_num]
        Xmin_f           = mask_info.item().get("Xmin_f")[target_num]
        Xmax_f           = mask_info.item().get("Xmax_f")[target_num]
        centroid_label_x = mask_info.item().get("centroid_label_x")[target_num]
        centroid_label_y = mask_info.item().get("centroid_label_y")[target_num]
        centroid_coord_y = centroid_label_y - Ymin_f
        centroid_coord_x = centroid_label_x - Xmin_f
        crop_image = img [Ymin_f:Ymax_f, Xmin_f:Xmax_f]
        crop_mask  = mask[Ymin_f:Ymax_f, Xmin_f:Xmax_f]

        #################################
        ## Add PIL Gaussian Noise
        crop_image_clean = crop_image.copy()
        try:
            crop_image_clean = img_as_float(Image.fromarray(crop_image_clean))
        except:
            print()
        if   Noise_type =='salt':
          crop_image_noise = util.random_noise(crop_image_clean, mode=Noise_type, amount=amount)
        elif Noise_type =='gaussian':
          crop_image_noise = util.random_noise(crop_image_clean, mode=Noise_type, var   =amount)
        elif Noise_type =='pepper':
          crop_image_noise = util.random_noise(crop_image_clean, mode=Noise_type, amount=amount)
        crop_image       = crop_image_noise
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(crop_image_clean)
        # plt.subplot(122)
        # plt.imshow(crop_image)
        # plt.show()
        #################################

        if   mask_info.item().get("target_type")[target_num] == 'Point':
            n_segments = segments_num[0]
            segments = slic(crop_image, n_segments=n_segments, compactness=20)
            # print('Point')
        elif mask_info.item().get("target_type")[target_num] == 'Spot':
            n_segments = segments_num[1]
            segments = slic(crop_image, n_segments=n_segments, compactness=20)
            # print('Spot')
        elif mask_info.item().get("target_type")[target_num] == 'Extended':
            n_segments = segments_num[2]
            segments = slic(crop_image, n_segments=n_segments, compactness=20)

        out_boundary = mark_boundaries(crop_image, segments)
        centroid_coord_y = (crop_image.shape[0]-1  if crop_image.shape[0] <= centroid_coord_y else int(round(centroid_coord_y)))
        centroid_coord_x = (crop_image.shape[1]-1  if crop_image.shape[1] <= centroid_coord_x else int(round(centroid_coord_x)))

        area_num     = segments[int(round(centroid_coord_y)), int(round(centroid_coord_x))]
        target_area  = np.where(segments == area_num)
        for point in range(np.array(target_area).shape[1]):
            WS_mask[Ymin_f + target_area[0][point], Xmin_f + target_area[1][point]] = 255
        crop_WS_mask = WS_mask[Ymin_f:Ymax_f, Xmin_f:Xmax_f]
    return  WS_mask, out_boundary, crop_image, crop_mask, crop_WS_mask, mask_info.item().get("target_type")

def generate_final(mask_info, img, Final_repeated_mask, Target_threshold):
    Final_result    = np.zeros_like(Final_repeated_mask)
    Final_result_T  = Final_result.copy()
    MODEL_NUM_CLASSES = 2
    CRF_ITER = 1
    for target_num in range(len(mask_info.item().get("Ymax_f"))):
        Ymin_f = mask_info.item().get("Ymin_f")[target_num]
        Ymax_f = mask_info.item().get("Ymax_f")[target_num]
        Xmin_f = mask_info.item().get("Xmin_f")[target_num]
        Xmax_f = mask_info.item().get("Xmax_f")[target_num]
        crop_image                          = img[Ymin_f:Ymax_f, Xmin_f:Xmax_f].transpose(2, 0, 1)
        Final_repeated_crop_mask            = Final_repeated_mask[Ymin_f:Ymax_f, Xmin_f:Xmax_f]
        Target_probability_map              = np.expand_dims(Final_repeated_crop_mask / Final_repeated_crop_mask.max(), axis=0)
        Background_probability_map          = 1 - Target_probability_map
        prob        = np.concatenate([Background_probability_map, Target_probability_map], axis=0)
        img_batched = crop_image
        prob        = dense_crf(prob, img_batched, n_classes=MODEL_NUM_CLASSES, n_iters=CRF_ITER)
        prob_seg    = prob.astype(np.float32)
        crop_result = np.argmax(prob_seg, axis=0)
        Final_result[Ymin_f:Ymax_f, Xmin_f:Xmax_f] = crop_result
        crop_mask   = mask[Ymin_f:Ymax_f, Xmin_f:Xmax_f]

        Target_probability_map_T = Target_probability_map.copy()
        Target_probability_map_T[Target_probability_map_T < Target_threshold] = 0
        Target_probability_map_T[Target_probability_map_T > 0]                = 255
        Final_result_T[Ymin_f:Ymax_f, Xmin_f:Xmax_f] = Target_probability_map_T

        Final_result[Final_result > 0]                = 255

    return  Final_result, crop_image, crop_mask, Target_probability_map, crop_result, Final_result_T



if __name__ == "__main__":
    dataset_root   =  '/media/gfkd/sda/Dataset'
    dataset        =  'NUAA-SIRST'

    ####CRF V1 V2
    save_dir          =  'Salt_with_size_prior_best'
    Event_name        =  '0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_01_11_2023_11_34_58_Retrain_2'
    final_save_dir    =  '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/'+ Event_name + '/' + 'MCLC_optimization_root/%s' % (save_dir)
    txt_save_dir      =  '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/'+ Event_name + '/' + 'MCLC_optimization_root'
    original_img_dir  =  '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/'+ Event_name + '/' + 'visulization_result'
    original_mask_dir =  dataset_root + '/' + dataset + '/' + 'images'

    ## CRF V1 V2
    mask_info_dir     = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/'+ Event_name + '/' + 'image_info'
    txt_dir           =  dataset_root + '/' + dataset + '/' + 'mode/' + 'test.txt'

    # SP_Num_list = np.arange(1,51)
    SP_Num_list = [9]
    for total_iter_num in range(1):
        for num_of_SP in range(len(SP_Num_list)):
            segments_num           = [0, 0, 0]
            Target_threshold       =  0.7
            Repeated_num           =  50
            Noise_type             =  'salt'
            Noise_intensity_amount =  0.05
            SP_num                 =  SP_Num_list[num_of_SP]


            if not os.path.exists(final_save_dir):
                os.makedirs(final_save_dir)

            img_id  = []
            with open(txt_dir, "r") as f:
                line = f.readline()
                while line:
                    img_id.append(line.split('\n')[0].split('.png')[0])
                    line = f.readline()
                f.close()


            for img_num in range(len(img_id)):
                selected_repeated_id = []
                id_num = img_id[img_num]
                print('id_num:',id_num)
                print('Ratio:',img_num, '/', len(img_id))

                mask_info = np.load(mask_info_dir + '/' + id_num + '.npy', allow_pickle=True)
                if mask_info.item().get("target_type") != 'non_target':

                    # print(img_num, '===>', id_num)
                    ## v1 v2
                    for m in range(Repeated_num):
                        selected_repeated_id.append(id_num)

                    ###########################
                    ### Overall evaluation
                    Final_repeated_mask   = np.zeros_like(np.array(Image.open(original_img_dir + '/' + id_num + '.png').convert('RGB'))[:,:,0],dtype=np.float32)
                    Final_result          = np.zeros_like(Final_repeated_mask)
                    Final_repeated_mask_n = Final_repeated_mask.copy()
                    Final_result_n        = Final_result.copy()
                    for num in range(len(selected_repeated_id)):

                        segments_num[0] = SP_num
                        segments_num[1] = SP_num
                        segments_num[2] = SP_num

                        img  = np.array(Image.open(original_img_dir  + '/' + selected_repeated_id[num]  + '.png' ).convert('RGB'))
                        mask = np.array(Image.open(original_mask_dir + '/' + selected_repeated_id[num]  + '.png' ).convert('RGB'))
                        ###################### Basic version
                        FinalMask,   out_boundary,   crop_image,   crop_mask,   crop_WS_mask,   target_type   = super_pixel                      (img, mask, mask_info, segments_num   )
                        FinalMask_n, out_boundary_n, crop_image_n, crop_mask_n, crop_WS_mask_n, target_type_n = super_pixel_noise_with_size_prior(img, mask, mask_info, segments_num, Noise_type, Noise_intensity_amount)
                        Final_repeated_mask   += FinalMask
                        Final_repeated_mask_n += FinalMask_n

                    # print('target_type:', target_type)

                    Final_result,   crop_image,   crop_mask,   Target_probability_map,   crop_result   , Final_result_T  = generate_final(mask_info, img, Final_repeated_mask  , Target_threshold)
                    Final_result_n, crop_image_n, crop_mask_n, Target_probability_map_n, crop_result_n , Final_result_T_n= generate_final(mask_info, img, Final_repeated_mask_n, Target_threshold)


                    if not os.path.exists(final_save_dir + '/' + 'with_type_wo_noise'):
                        os.makedirs(final_save_dir       + '/with_type_wo_noise')
                    cv2.imwrite(final_save_dir           + '/with_type_wo_noise'    + '/' + id_num.split('.png')[0] + '_MCLC.png', Final_result)

                    if not os.path.exists(final_save_dir + '/' + 'wo_type_add_noise_crf'):
                        os.makedirs(final_save_dir       + '/wo_type_add_noise_crf')
                    cv2.imwrite(final_save_dir           + '/wo_type_add_noise_crf' + '/' + id_num.split('.png')[0] + '_MCLC.png', Final_result_n)

                    if not os.path.exists(final_save_dir + '/' + 'wo_type_add_noise_T'):
                        os.makedirs(final_save_dir       + '/wo_type_add_noise_T')
                    cv2.imwrite(final_save_dir           + '/wo_type_add_noise_T' + '/' + id_num.split('.png')[0] + '_MCLC.png',   Final_result_T_n)
                else:
                    Final_result     = np.zeros_like(np.array(Image.open(original_img_dir + '/' + id_num + '.png').convert('RGB'))[:,:,0],dtype=np.float32)
                    Final_result_n   = Final_result
                    Final_result_T_n = Final_result
                    if not os.path.exists(final_save_dir + '/' + 'with_type_wo_noise'):
                        os.makedirs(final_save_dir       + '/with_type_wo_noise')
                    cv2.imwrite(final_save_dir           + '/with_type_wo_noise'    + '/' + id_num.split('.png')[0] + '_MCLC.png', Final_result)

                    if not os.path.exists(final_save_dir + '/' + 'wo_type_add_noise_crf'):
                        os.makedirs(final_save_dir       + '/wo_type_add_noise_crf')
                    cv2.imwrite(final_save_dir           + '/wo_type_add_noise_crf' + '/' + id_num.split('.png')[0] + '_MCLC.png', Final_result_n)

                    if not os.path.exists(final_save_dir + '/' + 'wo_type_add_noise_T'):
                        os.makedirs(final_save_dir       + '/wo_type_add_noise_T')
                    cv2.imwrite(final_save_dir           + '/wo_type_add_noise_T' + '/' + id_num.split('.png')[0] + '_MCLC.png',   Final_result_T_n)


            evaluation(txt_save_dir, save_dir, original_img_dir, Target_threshold, Repeated_num, Noise_type, Noise_intensity_amount, total_iter_num, SP_num)






