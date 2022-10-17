import os
import cv2
import shutil
import numpy as np
pred_root = '/home/liucl/PycharmProjects/nosupAD/UGTR-MI/result/poloy500uncert/pred'
ori_root = '/home/liucl/PycharmProjects/nosupAD/dataset/TestDataset'
save_root = '/home/liucl/PycharmProjects/nosupAD/poloy-pred'
method_name = 'UGTR'
gt=False
for sub_dir in os.listdir(ori_root):
    if sub_dir =='test':
        continue
    for img_name in os.listdir(os.path.join(ori_root, sub_dir,'images')):

        ori_path = os.path.join(ori_root, sub_dir,'images', img_name)
        if gt :
            pred_path = os.path.join(ori_root, sub_dir,'masks', img_name)
        else:
            pred_path = os.path.join(pred_root, sub_dir, img_name)
        save_dir = os.path.join(save_root, method_name, sub_dir)
        os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_root, method_name, sub_dir, img_name)

        pred_img = cv2.imread(pred_path)

        ori_img = cv2.imread(ori_path)
        if pred_img is None:
            pred_img = np.zeros_like(ori_img)
        if pred_img is None or ori_img is None:
            print(pred_path,ori_path)
        pred_img = cv2.resize(pred_img,(ori_img.shape[1],ori_img.shape[0]))

        heatmap = cv2.applyColorMap(pred_img, cv2.COLORMAP_JET)
        add_img = cv2.addWeighted(ori_img, 0.7, heatmap, 0.5, 0)
        cv2.imwrite(save_path, add_img)
