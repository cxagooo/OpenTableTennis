import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from OpenAndPick import pick
from src import model
from src import util
from src.body import Body
import os

def detect(path, output, outpath, output_folder, filename):
    # hand_estimation = Hand('model/hand_pose_model.pth')
    body_estimation = Body('model/body_pose_model.pth')
    device = 'cuda:0'  # 假设你使用的是第一个 GPU
    body_estimation = body_estimation.to(device)
    test_image = path
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    print(0)
    canvas = copy.deepcopy(oriImg)
    # detect hand
    '''hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    '''
    with open(output, 'w') as f:
        f.write(str(candidate))
    pick(output, outpath, 3, 4, 10, 1)
    print(candidate)
    #保存图片
    plt.imsave(os.path.join(output_folder, filename), canvas[:, :, [2, 1, 0]])
    #plt.imsave(canvas[:, :, [2, 1, 0]])
    # plt.axis('off')0
    # plt.show()
    # return print("success")
detect('CutFrame_Output/output1/frame_0.png',1,1,1,1)