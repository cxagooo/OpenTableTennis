import time
import pandas as pd
from demo1 import detect
from OpenAndPick import pick
from cut import split_video_into_parts
import numpy as np
from utils import get_data
from src.body import Body
from threading import Thread
import torch
from concurrent.futures import ThreadPoolExecutor
import os
import csv
def data_change (data):
    # 将数据转换为 DataFrame
    # 这里我们假设数据中有 149 个序列，每个序列有 7 个时间步，每个时间步有 3 个特征，每个特征有 2 个数值
    # 因此，data 的形状为 (149, 7, 3, 2)
    sequences, sequence_length, features, values_per_feature = data.shape
    # 将数据重塑为 (sequences * sequence_length, features * values_per_feature)
    reshaped_data = data.reshape(-1, features * values_per_feature)
    # 创建 DataFrame
    columns = [f'Feature_{i}_{j}' for i in range(features) for j in range(values_per_feature)]
    df = pd.DataFrame(reshaped_data, columns=columns)
    # 添加序列和时间步索引
    df['Sequence'] = np.repeat(np.arange(sequences), sequence_length)
    df['TimeStep'] = np.tile(np.arange(sequence_length), sequences)
    # 重新排序列
    df = df[['Sequence', 'TimeStep'] + columns]
    # 保存为 CSV 文件
    df.to_csv('data.csv', index=False)



def txt_to_csv(path, number_of_dir, number_of_use):


    # 目录路径
    base_dir = path
    output_csv_file = 'all_data.csv'

    # 创建或清空输出文件
    with open(output_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # 写入表头
        csvwriter.writerow(['Output Folder', 'Use File', 'Data'])

    # 遍历所有的output文件夹
    for output_folder in range(number_of_dir):
        folder_path = os.path.join(base_dir, f'output{output_folder}')
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist.")
            continue

        # 遍历use文件
        for use_file in range(number_of_use):
            txt_file_path = os.path.join(folder_path, f'use{use_file}.txt')

            # 检查文件是否存在
            if not os.path.isfile(txt_file_path):
                print(f"File {txt_file_path} does not exist.")
                continue

            # 读取txt文件内容并追加到csv文件
            with open(txt_file_path, 'r') as txtfile:
                lines = txtfile.readlines()

            # 追加到csv文件
            with open(output_csv_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for line in lines:
                    row = line.strip().strip('[]').split()
                    row = [float(i) for i in row]
                    csvwriter.writerow([f'output{output_folder}', f'use{use_file}'] + row)

            print(f"Appended data from {txt_file_path} to {output_csv_file}")
def get_gpu_memory_info():
    # 获取GPU的总显存、已分配显存和缓存显存
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    cached_memory = torch.cuda.memory_cached(0)

    # 计算可用显存
    available_memory = total_memory - allocated_memory - cached_memory
    return available_memory

def cutter(number_of_output, frames) :
    for i in range(number_of_output):
        #os.mkdir(f'CutFrame_Output/output{i}')
        #p = i + 24
        video_path = f'/home/cxgao/Videos/data3/31-标记-{i}.mp4'
        output_dir = f'CutFrame_Output/output{i}'
        split_video_into_parts(video_path, output_dir, frames)

def infer(i, frames,body_estimation):
    for j in range(frames):
        path = f'CutFrame_Output/output{i}/frame_{j}.png'
        print(path)
        output = f'CutFrame_Output/output{i}/tem.txt'
        output_path = f'CutFrame_Output/output{i}/use{j}.txt'
        detect(path, output, output_path, body_estimation=body_estimation)

def detect_frames(number_of_output, frames):
    body_estimation = Body('model/body_pose_model.pth')
    with ThreadPoolExecutor(max_workers=5) as executor:
        # executor.map(infer, range(number_of_output))
        tasks = [executor.submit(infer, i, frames, body_estimation) for i in range(number_of_output)]
        print(tasks)
    # for i in range(number_of_output):
    #     while get_gpu_memory_info() < 50000:
    #         time.sleep(1)
    #     Thread(target=infer, args=(i,frames,body_estimation)).start()

    # data =get_data('CutFrame_Output')
def verify (number, frames):
    for i in range(number):
        for j in range(frames):
            # 文件路径
            file_path = f'CutFrame_Output/output{i}/use{j}.txt' # 替换为你的文件路径

            # 读取文件并解析每一行
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 解析第三行的第2个数字
            third_line = lines[2].strip()  # 去除行尾的换行符
            third_line_elements = third_line.strip('[]').split()  # 去除方括号并按空格分割
            first_element_of_third_line = float(third_line_elements[1])  # 2一个元素转换为浮点数
            print(first_element_of_third_line)
            integer_value = int(first_element_of_third_line / 100)  # 除以100并取整

            # 检查该数字是否为1

            if integer_value != 10 :
                source = f'CutFrame_Output/output{i}/use{j}.txt'
                target = f'CutFrame_Output/output{i - 1}/use{j}.txt'
                with open(source, 'r') as source_file:
                    source_lines = source_file.readlines()
                # 读取目标文件
                with open(target, 'r') as target_file:
                    target_lines = target_file.readlines()
                # 替换第三行
                if len(target_lines) > 2:  # 确保目标文件至少有三行
                    source_lines[2] = target_lines[2]  # 替换第三行
                # 写回修改后的源文件
                with open(source, 'w') as source_file:
                    source_file.writelines(source_lines)
                print("Third line in", source, "has been replaced.")
                print(f'CutFrame_Output/output{i}/use{j}.txt')
def write_data (path):
    with open(path, 'w') as f:
        f.write(str(get_data('CutFrame_Output')))
if __name__ == '__main__':
    #verify(149, 7)
    #detect_frames(149, 7)
    #data = get_data('CutFrame_Output')
    #data_change(data)
    txt_to_csv("CutFrame_Output", 149, 7)



