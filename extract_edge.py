import os
import cv2
import numpy as np


def get_edge(labelroot,savedir):
    for label_name in os.listdir(labelroot):
        label_path = os.path.join(labelroot,label_name)
        edge_path = os.path.join(savedir,label_name)
        label_img = cv2.imread(label_path)
        edge = cv2.Canny(label_img, 50, 200)
        # cv2.imshow("color edge", edge)
        # cv2.waitKey(1)
        cv2.imwrite(edge_path, edge)
if __name__=='__main__':
    label_dir = '/home/liucl/PycharmProjects/nosupAD/indusData/TestDataset'

    for subdir in os.listdir(label_dir):
        labelroot = os.path.join(label_dir,subdir,'masks')
        savedir = os.path.join(label_dir,subdir,'edges')
        os.makedirs(savedir,exist_ok=True)
        get_edge(labelroot,savedir)