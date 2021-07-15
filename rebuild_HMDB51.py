import cv2
import matplotlib.pyplot as plt
import copy,os,sys
import numpy as np
from src import model
from src import util
from src.body import Body
base_path=sys.argv[1]#'/home/eed-server3/Desktop/DATA4T/mars/MARS-master/dataset/HMDB51'
processed_path=sys.argv[2]#'./HMDB51'
body_estimation = Body('model/body_pose_model.pth')
a=os.listdir('./HMDB51_labels')
a.sort()
for hmdb51 in a:
    if 'split1' in hmdb51:
        with open(os.path.join('./HMDB51_labels',hmdb51),'r+') as f:
            frame_folder = f.readline().split('.')[0]
            while frame_folder:
                frame_folder_all = os.path.join(base_path,hmdb51.split('_train')[0].split('_test')[0],frame_folder)
                for frame_name in os.listdir(frame_folder_all):
                    if '_' not in frame_name and '.jpg' in frame_name:
                        #print(frame_name)
                        oriImg = cv2.imread(os.path.join(frame_folder_all,frame_name))  # B,G,R order
                        candidate, subset = body_estimation(oriImg)
                        canvas = copy.deepcopy(oriImg)
                        canvas = util.draw_bodypose(canvas, candidate, subset)
                        tmp_dir=os.path.join(processed_path,hmdb51.split('_train')[0].split('_test')[0],frame_folder)
                        if not os.path.exists(tmp_dir):
                            os.makedirs(tmp_dir)
                        cv2.imwrite(os.path.join(tmp_dir,frame_name),canvas)
                frame_folder = f.readline().split('.')[0]

