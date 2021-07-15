import cv2
import matplotlib.pyplot as plt
import copy,os
import numpy as np
from src import model
from src import util
from src.body import Body
base_path=sys.argv[1]#'/DATA4T/mars/MARS-master/dataset/UCF101'
processed_path=sys.argv[2]#'./UCF101'
body_estimation = Body('model/body_pose_model.pth')
with open('label_ucf.txt','r+') as f:
    frame_folder = f.readline().split('.')[0]
    while frame_folder:
        frame_folder_all = os.path.join(base_path,frame_folder)
        for frame_name in os.listdir(frame_folder_all):
            if '_' not in frame_name and '.jpg' in frame_name:
                oriImg = cv2.imread(os.path.join(frame_folder_all,frame_name))  # B,G,R order
                candidate, subset = body_estimation(oriImg)
                canvas = copy.deepcopy(oriImg)
                canvas = util.draw_bodypose(canvas, candidate, subset)
                tmp_dir=os.path.join(processed_path,frame_folder)
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                cv2.imwrite(os.path.join(tmp_dir,frame_name),canvas)
        frame_folder = f.readline().split('.')[0]

