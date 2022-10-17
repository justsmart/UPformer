import os,shutil


ori_root = '/home/liucl/PycharmProjects/nosupAD/dataset/TestDataset'
pred_root = '/home/liucl/PycharmProjects/nosupAD/poloy-pred'

image_name = 'cju7dymur2od30755eg8yv2ht.png'
class_name = 'Kvasir'
save_dir='/home/liucl/PycharmProjects/nosupAD/poloy-selected/13'

os.makedirs(save_dir,exist_ok=True)
for sub_m in os.listdir(pred_root):
    sub_c = class_name
    for sub_img in os.listdir(os.path.join(pred_root, sub_m,sub_c)):
        if image_name==sub_img:
            shutil.copy(os.path.join(pred_root, sub_m,sub_c,sub_img),os.path.join(save_dir,sub_m+'.png'))
            shutil.copy(os.path.join(ori_root, sub_c, 'images', image_name),
                        os.path.join(save_dir, image_name))
            break
