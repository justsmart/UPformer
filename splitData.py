import os

test_list = ['test']
for testdata in test_list:
    img_root = f'/home/liucl/PycharmProjects/nosupAD/dataset/TestDataset/{testdata}/images'
    mask_root = f'/home/liucl/PycharmProjects/nosupAD/dataset/TestDataset/{testdata}/masks'

    file_path =f'./data/{testdata}-test.list'
    if os.path.exists(file_path):
            os.remove(file_path)
    with open(file_path,'a') as f:
        for image in os.listdir(img_root):
            img_path = os.path.join(img_root,image)
            mask_path = os.path.join(mask_root,image)

            f.write(img_path+' '+mask_path+'\n')
        f.close()
