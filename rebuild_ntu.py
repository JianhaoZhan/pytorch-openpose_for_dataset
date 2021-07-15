# python demo.py /New-8T/mars/NTU_RGB /New-8T/mars/NTU_Skeleton
import cv2,os,sys
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import model
from src import util
from src.body import Body
from src.hand import Hand
rgb_dir = sys.argv[1]
ske_dir = sys.argv[2]
l = ['nturgbd_rgb_s001', 'nturgbd_rgb_s002', 'nturgbd_rgb_s003', 'nturgbd_rgb_s004', 'nturgbd_rgb_s005', 'nturgbd_rgb_s006', 'nturgbd_rgb_s007', 'nturgbd_rgb_s008', 'nturgbd_rgb_s009', 'nturgbd_rgb_s010', 'nturgbd_rgb_s011', 'nturgbd_rgb_s012', 'nturgbd_rgb_s013', 'nturgbd_rgb_s014', 'nturgbd_rgb_s015', 'nturgbd_rgb_s016', 'nturgbd_rgb_s017']
ppp = []
for i in l:
    dirs = os.listdir(os.path.join(rgb_dir, i))
    dirs.sort()
    for j in dirs:
        ppp.append(os.path.join(rgb_dir, i, j))

body_estimation = Body('model/body_pose_model.pth')
#hand_estimation = Hand('model/hand_pose_model.pth')

for dirs in ppp: 
    res_d = os.path.join(ske_dir, '/'.join(dirs.split('/')[-2:]))
    if not os.path.exists(res_d):
        os.makedirs(res_d)
    for img in os.listdir(dirs):
        if img == 'done':
            continue
        test_image = os.path.join(dirs, img)
        write_image = os.path.join(ske_dir, '/'.join(test_image.split('/')[-3:]))
        if os.path.exists(write_image):
            continue
        oriImg = cv2.imread(test_image)  # B,G,R order
        oriImg = oriImg[:, 40:415]
        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        # detect hand
        # hands_list = util.handDetect(candidate, subset, oriImg)

        # all_hand_peaks = []
        # for x, y, w, is_left in hands_list:
        #     # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        #     # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #     # if is_left:
        #         # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        #         # plt.show()
        #     peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     # else:
        #     #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     #     print(peaks)
        #     all_hand_peaks.append(peaks)

        # canvas = util.draw_handpose(canvas, all_hand_peaks)
        #canvas = cv2.pyrDown(canvas)
        cv2.imwrite(write_image, canvas)
        
