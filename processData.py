import os
import shutil
import random
import time
rawroot  = '/home/liucl/PycharmProjects/nosupAD/indus_raw_data'
saveroot = '/home/liucl/PycharmProjects/nosupAD/indusData'
save_Train_path = os.path.join(saveroot,'TrainDataset')
save_Val_path = os.path.join(saveroot,'ValDataset')
save_Test_path = os.path.join(saveroot,'TestDataset')
for bigclass in os.listdir(rawroot):
    bigclass_path = os.path.join(rawroot,bigclass)
    gt_path = os.path.join(bigclass_path,'ground_truth')
    image_path = os.path.join(bigclass_path, 'test')
    os.makedirs(os.path.join(save_Train_path, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(save_Train_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_Val_path, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(save_Val_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_Test_path,bigclass,'masks'),exist_ok=True)
    os.makedirs(os.path.join(save_Test_path, bigclass, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_Test_path, 'test','masks'), exist_ok=True)
    os.makedirs(os.path.join(save_Test_path, 'test', 'images'), exist_ok=True)
    random.seed(10)
    for i,subclass in enumerate(os.listdir(gt_path)):
        subgt_path = os.path.join(gt_path, subclass)
        subimg_path = os.path.join(image_path, subclass)
        img_list=os.listdir(subgt_path)
        sub_len = len(img_list)
        random.shuffle(img_list)
        train_list = img_list[:int(sub_len*0.8)]
        test_list = img_list[int(sub_len*0.8):]

        for gt_name in train_list:
            if '._000_' in gt_name:
                continue
            ori_img_name=gt_name.replace('_mask', '')

            img_name = bigclass+'-'+subclass+'-'+ori_img_name
            ori_gt_path = os.path.join(subgt_path,gt_name)
            new_gt_path = os.path.join(save_Train_path,'masks',img_name)
            shutil.copy(ori_gt_path,new_gt_path)
            ori_img_path = os.path.join(subimg_path, ori_img_name)
            new_img_path = os.path.join(save_Train_path, 'images', img_name)
            shutil.copy(ori_img_path, new_img_path)

        for gt_name in test_list:
            if '._000_' in gt_name:
                continue
            ori_img_name = gt_name.replace('_mask', '')
            img_name = bigclass+'_'+subclass+'_'+ori_img_name
            ori_gt_path = os.path.join(subgt_path, gt_name)
            new_gt_path1 = os.path.join(save_Test_path,bigclass, 'masks', img_name)
            new_gt_path2 = os.path.join(save_Test_path,'test', 'masks', img_name)
            shutil.copy(ori_gt_path, new_gt_path1)
            shutil.copy(ori_gt_path, new_gt_path2)
            ori_img_path = os.path.join(subimg_path, ori_img_name)
            new_img_path1 = os.path.join(save_Test_path,bigclass, 'images', img_name)
            new_img_path2 = os.path.join(save_Test_path,'test', 'images', img_name)
            shutil.copy(ori_img_path, new_img_path1)
            shutil.copy(ori_img_path, new_img_path2)
        if i in range(0,len(os.listdir(gt_path)),4):
            for gt_name in test_list:
                if '._000_' in gt_name:
                    continue
                ori_img_name=gt_name.replace('_mask', '')
                img_name = bigclass+'-'+subclass+'-'+ori_img_name
                ori_gt_path = os.path.join(subgt_path,gt_name)
                new_gt_path = os.path.join(save_Val_path,'masks',img_name)
                shutil.copy(ori_gt_path,new_gt_path)
                ori_img_path = os.path.join(subimg_path, ori_img_name)
                new_img_path = os.path.join(save_Val_path, 'images', img_name)
                shutil.copy(ori_img_path, new_img_path)
                break


