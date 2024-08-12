from sklearn.model_selection import train_test_split
import re
import glob
import numpy as np

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=None):
    """
    将数据集分割为训练集、验证集和测试集。

    参数:
    - X: 特征数据 (numpy array, pandas DataFrame, etc.)
    - y: 标签数据 (numpy array, pandas Series, etc.)
    - test_size: 测试集所占的比例 (float, 默认为0.2)
    - val_size: 验证集所占的比例 (float, 默认为0.2)
    - random_state: 随机种子 (int, 默认为None)

    返回:
    - X_train: 训练集的特征数据
    - X_val: 验证集的特征数据
    - X_test: 测试集的特征数据
    - y_train: 训练集的标签数据
    - y_val: 验证集的标签数据
    - y_test: 测试集的标签数据
    """
    # 首先将数据集分割为训练集+验证集 (80%) 和 测试集 (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 再次将训练集+验证集分割为训练集 (80% of 80% = 64%) 和 验证集 (20% of 80% = 16%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                      random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

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

def restore_changes(data, index):
    result_data = []

    # 定义文件路径
    input_file_path = f'CutFrame_Output/output{index}'  # 请替换为您的输入文件路径
    for j,i in enumerate(data):
        frame_data = []
        base = process_data(f'{input_file_path}/use{j}.txt')[3]
        for row in i:
            frame_data.append([x + y for x, y in zip(row, base)])
        result_data.append(frame_data)
    return result_data


