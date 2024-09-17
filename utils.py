from sklearn.model_selection import train_test_split
import re
import glob
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import cv2
import pandas as pd

def split_dataset(X, y, test_val_size=0.2, random_state=None):
    train_x = X[:int(len(X)*(1-test_val_size))]
    train_y = y[:int(len(y)*(1-test_val_size))]
    test_val_x = X[int(len(X)*(1-test_val_size)):]
    test_val_y = y[int(len(y)*(1-test_val_size)):]
    test_x = test_val_x[:int(len(test_val_x)*0.5)]
    test_y = test_val_y[:int(len(test_val_y)*0.5)]
    val_x = test_val_x[int(len(test_val_x)*0.5):]
    val_y = test_val_y[int(len(test_val_y)*0.5):]
    return train_x, val_x, test_x, train_y, val_y, test_y

def get_data(d: str) -> np.array:
    data = []
    for _dir in glob.glob(f'{d}/output*/'):
        data.append([list_change(f) for f in glob.glob(_dir + 'use*.txt')])
    data = np.array(data)
    return data

def process_data(input_file_path):
    # 创建一个空列表来存储处理后的数据
    processed_data = []
    with open(input_file_path, 'r') as file:
        for line in file:
            # 使用正则表达式移除方括号
            clean_line = re.sub(r'\[|\]', '', line)
            # 分割字符串，得到每个元素
            elements = clean_line.split()
            # 只保留前两个元素
            first_two_elements = elements[:2]
            # 将字符串转换为浮点数，并添加到结果列表
            processed_data.append([float(element) for element in first_two_elements])
    return processed_data

def list_change(i):
    # 定义文件路径
    input_file_path = i  # 请替换为您的输入文件路径

    processed_data = process_data(input_file_path)

    # 获取前三行的数据
    first_three_rows = processed_data[:3]
    # 获取第四行的数据
    fourth_row = processed_data[3]

    # 减去第四行的值
    result_data = []
    for row in first_three_rows:
        new_row = [x - y for x, y in zip(row, fourth_row)]
        result_data.append(new_row)

    return result_data

def restore_changes(data, index, input_file_path=None):
    data = data.reshape(-1, 3, 2)
    result_data = []

    if not input_file_path:
        # 定义文件路径
        input_file_path = f'CutFrame_Output/output{index}'  # 请替换为您的输入文件路径
    for j,i in enumerate(data):
        frame_data = []
        base = process_data(f'{input_file_path}/use{j}.txt')[3]
        for row in i:
            frame_data.append([x + y for x, y in zip(row, base)])
        result_data.append(frame_data)
    return result_data


def create_gaussian_kernel(kernel_size, sigma):
    x = torch.arange(kernel_size).float() - kernel_size // 2
    xx, yy = torch.meshgrid(x, x)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def gpu_gaussian_filter_torch(image, sigma, kernel_size=5):
    # 将输入图像转换为torch张量并移动到GPU
    # image_gpu = torch.from_numpy(image).float().to('cuda')
    image_gpu = image.float()
    if len(image.shape) == 2:  # 单通道图像
        image_gpu = image_gpu.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:  # 多通道图像
        image_gpu = image_gpu.permute(2, 0, 1).unsqueeze(0)

    # 创建高斯核
    kernel = create_gaussian_kernel(kernel_size, sigma).to('cuda')

    # 应用卷积
    filtered_image = F.conv2d(image_gpu, kernel, padding=kernel_size // 2)

    # 将结果转换回NumPy数组
    # filtered_image_cpu = filtered_image.squeeze().cpu().numpy()
    filtered_image_cpu = filtered_image.squeeze()
    return filtered_image_cpu

def replace_data(d: str) -> np.array:
    for _dir in glob.glob(f'{d}/output*/'):
        [replace_last_line_with_zeros(f) for f in glob.glob(_dir + 'use*.txt')]

def replace_last_line_with_zeros(file_path):
    # 读取文件中的所有数据
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 将最后一行替换为零
    if lines:
        # last_line = np.zeros_like(np.fromstring(lines[-1], sep=' '))
        last_line = np.array([[0, 0, 0, 0]])
        lines[-1] = ' '.join(map(str, last_line)) + '\n'

    # 将数据写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)

def save_obj(obj, name):
    with open(f'{name}.pkl', 'wb') as f:
        # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(obj, f, protocol=4)

def load_obj(name):
    with open(f'{name}.pkl', 'rb') as f:
        return pickle.load(f)

def pointing(index, data):
    points_ori = restore_changes(data[index], index, None)
    for n, f in enumerate(points_ori):
        img = cv2.imread(f'./CutFrame_Output/output{index}/frame_{n}.png')
        for q, p in enumerate(points_ori[n]):
            print((p[0], p[1]))
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
            cv2.putText(img, str(q), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(f'ori_visualized/output{index}_{n}.png', img)

# def correcting():
#     a=[]
#     files = glob.glob('./output_new/*')
#     for i in files:
#         match = re.match(r'./output_new/output(\d+)_(\d+).json', i)
#         a.append((int(match.group(1)), int(match.group(2))))
#     b = [pd.read_json(i)['content'][0] for i in files]
#     for j,i in enumerate(a):
#         print(j)
#         print(i)
#         with open(f'./CutFrame_Output/output{i[0]}/use{i[1]}.txt', 'r+') as f:
#             lines = f.readlines()
#             f.truncate(0)
#             print(b[0])
#             lines0 = [[b[j][k]['x'],b[j][k]['y'],0,0] for k in range(0,len(b[j]))]
#             print(lines0)
#             for k in range(0,len(b[j])):
#                 print(lines)
#                 lines[k] = ' '.join(str(np.array(lines0[k])))
#                 print(lines[k])
#             f.writelines(lines)
def correcting():
    a=[]
    files = glob.glob('./output_new/*')
    for i in files:
        match = re.match(r'./output_new/output(\d+)_(\d+).json', i)
        a.append((int(match.group(1)), int(match.group(2))))
    b = [pd.read_json(i)['content'][0] for i in files]
    for j,i in enumerate(a):
        with open(f'./CutFrame_Output/output{i[0]}/use{i[1]}.txt', 'w') as f:
            lines0 = [[b[j][k]['x'],b[j][k]['y'],0,0] for k in range(0,len(b[j]))]
            lines0.append([0,0,0,0])
            lines = [str(np.array(i))+'\n' for i in lines0]
            f.writelines(lines)

def readCsv():
    a = np.delete(np.array(pd.read_csv("all_data.csv")), 0, axis=1)
    b = np.delete(a, 0, axis=1)
    b = b.reshape(int(len(b) / 7), 7, 6)
    return b.astype(np.float16)
