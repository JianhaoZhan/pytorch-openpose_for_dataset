import cv2
import matplotlib.pyplot as plt
import copy,os
import numpy as np
from src import model
from src import util
from src.body import Body
base_path='/DATA4T/mars/MARS-master/dataset/UCF101'
res_path='./result_ucf'
processed_path='./UCF101'
body_estimation = Body('model/body_pose_model.pth')
with open('list2.txt','r+') as f:
    frame_folder = f.readline().split('.')[0]
    #os.makedirs(os.path.join(res_path,frame_folder.split('/')[0]))
    # with open(os.path.join(res_path,frame_folder+'.skeleton'),'a+') as f1:
    #f1.write(str((len(os.listdir(frame_folder_all))+1)//3))
    #f1.write('\n')
    while frame_folder:
        #f1.write('72057594037931101 0 1 1 1 1 0 0.03022554 0.04645365 2\n')
        frame_folder_all = os.path.join(base_path,frame_folder)
        for frame_name in os.listdir(frame_folder_all):
            if '_' not in frame_name and '.jpg' in frame_name:
                #print(frame_name)
                oriImg = cv2.imread(os.path.join(frame_folder_all,frame_name))  # B,G,R order
                candidate, subset = body_estimation(oriImg)
                canvas = copy.deepcopy(oriImg)
                canvas = util.draw_bodypose(canvas, candidate, subset)
                tmp_dir=os.path.join(processed_path,frame_folder)
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                cv2.imwrite(os.path.join(tmp_dir,frame_name),canvas)
        frame_folder = f.readline().split('.')[0]

