from sklearn.model_selection import train_test_split
import re
import glob
import numpy as np
import torch
import torch.nn.functional as F


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

