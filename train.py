import time

from demo1 import detect
from OpenAndPick import pick
from cut import split_video_into_parts
import os
from utils import get_data
from src.body import Body
from threading import Thread
import torch
from concurrent.futures import ThreadPoolExecutor


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

            # 解析第三行的第一个数字
            third_line = lines[2].strip()  # 去除行尾的换行符
            third_line_elements = third_line.strip('[]').split()  # 去除方括号并按空格分割
            first_element_of_third_line = float(third_line_elements[0])  # 第一个元素转换为浮点数
            integer_value = int(first_element_of_third_line / 100)  # 除以100并取整

            # 检查该数字是否为5

            if integer_value != 5 :
                if integer_value != 4:
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
if __name__ == '__main__':
    verify(149, 7)
    #detect_frames(149, 7)



