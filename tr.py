import os,shutil
root = '/home/liucl/PycharmProjects/nosupAD/map/UNet/CVC-ColonDB'
for img in os.listdir(root):
    if img.split('.')[-1]=='tif':
        os.rename(os.path.join(root,img),os.path.join(root,img.replace('tif','png')))