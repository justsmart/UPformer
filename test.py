import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize, calc_mae, check_makedirs,calc_dic
import pdb
from util.dataloader import get_loader
import datetime
from PIL import Image

import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cod_resnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/cod_resnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def check(args):
    assert args.classes == 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if args.arch == 'ugtr':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    global args, logger
    args = get_parser()
    check(args)
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    date_str = str(datetime.datetime.now().date())
    save_folder = args.save_folder + '/' + date_str
    check_makedirs(save_folder)
    if args.new == True:
        from model.ugtr_new import UGTRNet
        print('model new!')
    else:
        from model.ugtr_ori import UGTRNet
        print('model ori!')

    # test_transform = transform.Compose([
    #          transform.Resize((args.test_h, args.test_w)),
    #          transform.ToTensor(),
    #           transform.Normalize(mean=mean, std=std)])

    # test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    
    # index_start = args.index_start
    # if args.index_step == 0:
    #     index_end = len(test_data.data_list)
    # else:
    #     index_end = min(index_start + args.index_step, len(test_data.data_list))
    # data_list = test_data.data_list[index_start:index_end]
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_names=[]
    for test_dir in os.listdir(args.test_root):
        if test_dir != '':
            test_names.append(test_dir)
    results = {}

    model = UGTRNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False, args=args,T=16,K=50)
    #logger.info(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    res_file=os.path.join(save_folder,'result.txt')
    ########## select fixed model
    # model_path = os.path.join(args.model_root,'train_epoch_500.pth')
    # if os.path.isfile(model_path):
    #     logger.info("=> loading checkpoint '{}'".format(model_path))
    #     checkpoint = torch.load(model_path, map_location='cuda:0')
    #
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     logger.info("=> loaded checkpoint '{}', epoch {}".format(model_path, checkpoint['epoch']))
    # else:
    #     raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
    ##########
    for test_data_name in test_names:
        ###### Iterate over the optimal model for each dataset
        model_path = os.path.join(args.model_root,'best_'+test_data_name+'.pth')
        if os.path.isfile(model_path):
            logger.info("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location='cuda:0')
        #
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}', epoch {}".format(model_path, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
        ######
        test_loader = get_loader(f'{args.test_root}/{test_data_name}/images',f'{args.test_root}/{test_data_name}/masks', 1,
                            args.train_h, shuffle=False, num_workers=0, pin_memory=True, augmentation=False,train=False)

        colors = np.loadtxt(args.colors_path).astype('uint8')
        names = [line.rstrip('\n') for line in open(args.names_path)]
        gray_folder = os.path.join(save_folder, 'pred',test_data_name)
        
        
        res_dict=test(test_loader, model, gray_folder)
        results[test_data_name]=res_dict
    if os.path.exists(res_file):
        os.remove(res_file)
    f = open(res_file,'a')
    for key in results.keys():
        res = results[key]
        logger.info('Test {}: mae / dice: {:.3f}/{:.3f}'.format(key,res['mae'], res['dice']))
        f.write('Test {}: mae / dice: {:.3f}/{:.3f}\n'.format(key,res['mae'], res['dice']))


def test(test_loader, model, gray_folder):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    mae = AverageMeter()
    dice = AverageMeter()
    model.eval()
    end = time.time()
    check_makedirs(gray_folder)
    for i, (input,target,image_name,image_BGR) in enumerate(test_loader):
        input = input.cuda(non_blocking=True)
        
        with torch.no_grad():
            pred, uncertainty, mean,vis_att = model(input)
            
        # region = torch.sigmoid(region)

        mean = torch.sigmoid(mean)
        uncertainty = torch.sigmoid(uncertainty)
        pred = F.interpolate(pred, size=target.size()[2:], mode='bilinear', align_corners=True)
        pred = pred.detach().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        if vis_att is not None:
            vis_att = F.interpolate(vis_att, size=target.size()[2:], mode='bilinear', align_corners=True)
            vis_att = vis_att.detach().cpu().numpy()
            
        # P_b = F.interpolate(P_b, size=target.size()[2:], mode='bilinear', align_corners=True)
        # P_b = P_b.detach().cpu().numpy()
        # P_b = (P_b - P_b.min()) / (P_b.max() - P_b.min())
        # P_f = F.interpolate(P_f, size=target.size()[2:], mode='bilinear', align_corners=True)
        # P_f = P_f.detach().cpu().numpy()
        # P_f = (P_f - P_f.min()) / (P_f.max() - P_f.min())
        mean = F.interpolate(mean, size=target.size()[2:], mode='bilinear', align_corners=True)
        mean = mean.detach().cpu().numpy()
        mean = (mean - mean.min()) / (mean.max() - mean.min())
        uncertainty = F.interpolate(uncertainty, size=target.size()[2:], mode='bilinear', align_corners=True)
        uncertainty = uncertainty.detach().cpu().numpy()
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        target = target.numpy()
        target /= (target.max())
        
        for j in range(len(pred)):
            dice.update(calc_dic(pred[j],target[j]))
            mae.update(calc_mae(pred[j],target[j]))
            pred_j = np.uint8(pred[j]*255)
            target_j = np.uint8(target[j]*255)
            # if pred_j is not None:
            ori_img = image_BGR[j].numpy()
            heatmap = cv2.applyColorMap(target_j.transpose(1,2,0), cv2.COLORMAP_JET)
            img_path = os.path.join(gray_folder, image_name[0] + '.png')
            add_img = cv2.addWeighted(ori_img, 0.7, heatmap, 0.5, 0)
            cv2.imwrite(img_path,pred_j.transpose(1,2,0))
            # cv2.imwrite(img_path, add_img)
        if mean is not None and uncertainty is not None:
            for j in range(len(mean)):
                mean_j = np.uint8(mean[j]*255)
                uncertainty_j = np.uint8(uncertainty[j]*255)

                img_path = os.path.join(gray_folder, image_name[0] + '_mean.png')
                # cv2.imwrite(img_path,mean_j.transpose(1,2,0))
                img_path = os.path.join(gray_folder, image_name[0] + '_un.png')
                heatmap = cv2.applyColorMap(uncertainty_j.transpose(1, 2, 0), cv2.COLORMAP_JET)
                # add_img = cv2.addWeighted(ori_img, 0.7, heatmap, 0.5, 0)
                # cv2.imwrite(img_path,heatmap)
        if vis_att is not None:
            for j in range(len(vis_att)):
                vis_att_j=vis_att[j]
                
                ori_img=image_BGR[j].numpy()

                for k in range(len(vis_att_j)):
                    vis_att_gray=vis_att_j[k] # (1, h, w)
                    vis_att_gray = (vis_att_gray - vis_att_gray.min()) / (vis_att_gray.max() - vis_att_gray.min())
                    # vis_att_gray = np.uint8(vis_att_gray*255)
                    vis_att_gray = torch.from_numpy(np.uint8(vis_att_gray * 255)).unsqueeze(0).numpy()
                    img_path = os.path.join(gray_folder, image_name[j] + '_{}.png'.format(k))
                    
                    # print(ori_img.shape,heatmap.shape)
                    heatmap = cv2.applyColorMap(vis_att_gray.transpose(1,2,0), cv2.COLORMAP_JET)
                    add_img = cv2.addWeighted(ori_img, 0.7, heatmap, 0.5, 0)
                    # print(heatmap.shape,heatmap)
                    # cv2.imwrite(img_path,vis_att_gray.transpose(1,2,0))
                    cv2.imwrite(img_path, add_img)
        # if ((i + 1) % 30 == 0) or (i + 1 == len(test_loader)):
        #     logger.info('Test: [{}/{}] '.format(i + 1, len(test_loader)))
        
        # time.sleep(3)     
    
        
        # cv2.imwrite(gray_path, gray)
    logger.info('val result: mae: {:.7f} // dice: {:.7f}'.format(mae.avg, dice.avg))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return {"mae":mae.avg,"dice":dice.avg}


def calc_acc(data_list, pred_folder):
    r_mae = AverageMeter()
    e_mae = AverageMeter()

    for i, (image_path, target1_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred1 = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)

        target1 = cv2.imread(target1_path, cv2.IMREAD_GRAYSCALE)

        if pred1.shape[0] != target1.shape[0] or pred1.shape[1] != target1.shape[1]:
            pred1 = cv2.resize(pred1, (target1.shape[1], target1.shape[0]))

        r_mae.update(calc_mae(pred1, target1))

        logger.info('Evaluating {0}/{1} on image {2}, mae {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', r_mae.avg))

    logger.info('Test result: r_mae / e_mae: {0:.3f}/{1:.3f}'.format(r_mae.avg, e_mae.avg))

if __name__ == '__main__':
    main()
