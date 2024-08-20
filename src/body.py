import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import transforms

from src import util
from src.model import bodypose_model
from utils import gpu_gaussian_filter_torch


# from cupy import from_dlpack
# from cupy.core import toDlpack
# import cupy as np
import numpy as np
# input("Press Enter to continue...")


import torch
import torch.nn.functional as F


def convert_heatmap(Mconv7_stage6_L2,a):
    # 确保输入张量的维度为 [batch_size, channels, height, width]
    if Mconv7_stage6_L2.dim() == 4 and Mconv7_stage6_L2.size(0) == 1:
        # 移除 batch 维度
        Mconv7_stage6_L2 = Mconv7_stage6_L2.squeeze(0)

    # 调整轴的顺序
    heatmap = Mconv7_stage6_L2.permute(1, 2, 0)

    return heatmap


def resize_tensor(input_tensor, size=(0,0), fx=1, fy=1, interpolation=cv2.INTER_LINEAR):
    # 确保输入张量的维度为 [batch_size, channels, height, width]
    if input_tensor.dim() == 3:
        pass  # 输入已经是正确的格式
    elif input_tensor.dim() == 4 and input_tensor.size(0) == 1:
        input_tensor = input_tensor.squeeze(0)  # 移除 batch 维度
    if size!=(0,0):
        input_tensor = input_tensor.permute(2, 0, 1)

        # 添加 batch 维度
        input_tensor = input_tensor.unsqueeze(0)

        # 使用 bilinear 插值（等同于 cv2.INTER_CUBIC 对于双线性插值）
        resized_tensor = F.interpolate(input_tensor, size=size, mode='bilinear', align_corners=False)

        # 移除 batch 维度
        resized_tensor = resized_tensor.squeeze(0)

        # 重新调整轴的顺序
        resized_tensor = resized_tensor.permute(2, 1, 0)
    else:
        print('b')
        resized_tensor = F.interpolate(input_tensor.permute(2, 0, 1).unsqueeze(0),
                                       scale_factor=fx,
                                       mode='bilinear',
                                       align_corners=False)
        # 重新调整轴的顺序
        resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)
        # resized_tensor = F.interpolate(input_tensor, scale_factor=fx, mode='bilinear', align_corners=False)


    return resized_tensor

class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()
        self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.5
        thre2 = 0.5
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        heatmap_avg = torch.tensor(heatmap_avg).cuda()
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        paf_avg = torch.tensor(paf_avg).cuda()

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            print(Mconv7_stage6_L1.shape)
            print(Mconv7_stage6_L2.shape)
            # input("Press Enter to continue...")
            # Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            # Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

            # extract outputs, resize, and remove padding
            # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
            heatmap = convert_heatmap(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps

            heatmap = resize_tensor(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)

            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = resize_tensor(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)


            # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            paf = convert_heatmap(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))  # output 0 is PAFs
            paf = resize_tensor(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = resize_tensor(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            print(paf.shape)
            print(paf_avg.shape)
            # input("Press Enter to continue...")
            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += + paf / len(multiplier)

        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            # one_heatmap = gaussian_filter(map_ori, sigma=3)  # 旧用法
            one_heatmap = gpu_gaussian_filter_torch(map_ori, sigma=3)

            map_left = torch.zeros_like(one_heatmap)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = torch.zeros_like(one_heatmap)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = torch.zeros_like(one_heatmap)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = torch.zeros_like(one_heatmap)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = (one_heatmap >= map_left) & (one_heatmap >= map_right) & (one_heatmap >= map_up) & (
                        one_heatmap >= map_down) & (one_heatmap > thre1)
            # peaks_binary = peaks_binary.cpu().numpy()
            peaks = list(zip(*torch.nonzero(peaks_binary, as_tuple=True)))  # note reverse
            peaks_with_score = [(x[1], x[0], map_ori[x[0], x[1]].item()) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correspondence
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = torch.tensor(candB[j][:2]).to('cuda') - torch.tensor(candA[i][:2]).to('cuda')
                        norm = torch.sqrt(torch.sum(vec ** 2))
                        norm = max(0.001, norm.item())
                        vec = vec / norm

                        startend = list(zip(torch.linspace(candA[i][0], candB[j][0], steps=mid_num),
                                            torch.linspace(candA[i][1], candB[j][1], steps=mid_num)))

                        vec_x = torch.tensor(
                            [score_mid[torch.round(startend[I][1]).long(), torch.round(startend[I][0]).long(), 0].item()
                             for I in range(mid_num)]).to('cuda')
                        vec_y = torch.tensor(
                            [score_mid[torch.round(startend[I][1]).long(), torch.round(startend[I][0]).long(), 1].item()
                             for I in range(mid_num)]).to('cuda')

                        score_midpts = vec_x * vec[0] + vec_y * vec[1]
                        score_with_dist_prior = torch.sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(torch.nonzero(score_midpts > thre2)) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior.item(),
                                 score_with_dist_prior.item() + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = torch.zeros((0, 5)).to('cuda')
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3]) and (j not in connection[:, 4]):
                        connection = torch.cat((connection, torch.tensor([[candA[i][3], candB[j][3], s, i, j]]).to('cuda')), dim=0)
                        if (connection.shape[0] >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append(torch.zeros((0, 5)).to('cuda'))

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * torch.ones((0, 20)).to('cuda')
        candidate = torch.tensor([item for sublist in all_peaks for item in sublist]).to('cuda')

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = torch.tensor(limbSeq[k]) - 1

                for i in range(connection_all[k].shape[0]):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(subset.shape[0]):  # 1:size(subset,1):
                        if subset[j, indexA] == partAs[i] or subset[j, indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j, indexB] != partBs[i]:
                            subset[j, indexB] = partBs[i]
                            subset[j, -1] += 1
                            subset[j, -2] += candidate[partBs[i].long(), 2] + connection_all[k][i, 2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).int() + (subset[j2] >= 0).int())[:-2]
                        if len(torch.nonzero(membership == 2)) == 0:  # merge
                            subset[j1, :-2] += (subset[j2, :-2] + 1)
                            subset[j1, -2:] += subset[j2, -2:]
                            subset[j1, -2] += connection_all[k][i, 2]
                            subset = torch.cat((subset[:j2], subset[j2 + 1:]), dim=0)
                        else:  # as like found == 1
                            subset[j1, indexB] = partBs[i]
                            subset[j1, -1] += 1
                            subset[j1, -2] += candidate[partBs[i].long(), 2] + connection_all[k][i, 2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * torch.ones(20).to('cuda')
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = torch.sum(candidate[connection_all[k][i, :2].long(), 2]) + connection_all[k][i, 2]
                        subset = torch.cat((subset, row.unsqueeze(0)), dim=0)
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(subset.shape[0]):
            if subset[i, -1] < 4 or subset[i, -2] / subset[i, -1] < 0.4:
                deleteIdx.append(i)
        # subset = torch.cat((subset[:min(deleteIdx)], subset[max(deleteIdx) + 1:]), dim=0)
        # Limit the number of detected people to `max_people`
        # max_people = 1  # You can set this to the desired maximum number of people
        # subset = subset[:max_people]

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        print(type(subset))
        print(subset)
        input("Press Enter to continue...")
        # print(candidate)
        return candidate.cpu().numpy(), subset.cpu().numpy()

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset

if __name__ == "__main__":
    body_estimation = Body('../model/body_pose_model.pth')

    test_image = '../CutFrame_Output/output0/frame_0.png'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(oriImg, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()
