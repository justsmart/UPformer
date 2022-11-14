import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, imgsize, augmentations,train):
        self.imgsize = imgsize
        self.augmentations = augmentations
        # print('augmentations:',self.augmentations)
        self.images = [os.path.join(image_root , f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root , f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.train=train
        if train:
            if self.augmentations == True:
                print('Using RandomRotation, RandomFlip')
                self.img_transform = transforms.Compose([
                    transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((self.imgsize, self.imgsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])
                self.gt_transform = transforms.Compose([ 
                    transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((self.imgsize, self.imgsize)),
                    transforms.ToTensor(),
                    ])

            else:
                print('no augmentation')
                self.img_transform = transforms.Compose([
                    transforms.Resize((self.imgsize, self.imgsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])

                self.gt_transform = transforms.Compose([
                    transforms.Resize((self.imgsize, self.imgsize)),
                    transforms.ToTensor()])
        else:
            # print('no train augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.imgsize, self.imgsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                # transforms.Resize((self.imgsize, self.imgsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image_BGR = self.bgr_loader(self.images[index])
        
        seed = np.random.randint(1234) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        # print(gt.size)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        if self.train:
            return image, gt
        else:

            name =self.images[index].split('/')[-1].split('.')[0]
            
            return image, gt,name,image_BGR

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
            else:
                print(img_path,'size is not equal to ',gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def bgr_loader(self, path):
        bgr = cv2.imread(path)
        return bgr

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.imgsize or w < self.imgsize:
            h = max(h, self.imgsize)
            w = max(w, self.imgsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, imgsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False,train=True):

    dataset = PolypDataset(image_root, gt_root, imgsize, augmentation,train=train)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)


    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root , f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root , f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
        self.len = len(self.images)

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
