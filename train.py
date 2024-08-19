from demo1 import detect
from OpenAndPick import pick
from cut import split_video_into_parts
import os
from utils import get_data
#截取帧
#for i in range(24):
#    #os.mkdir(f'CutFrame_Output/output{i}')
#    video_path = f'//home/cxgao/Videos/pyopen/12-标记-{i}.mp4'
#    output_dir = f'CutFrame_Output/output{i}'
#    adjust_and_extract_frames(video_path, output_dir, 8)
def cutter(number_of_output, frames) :
    for i in range(number_of_output):
        #os.mkdir(f'CutFrame_Output/output{i}')
        #p = i + 24
        video_path = f'/home/cxgao/Videos/data3/31-标记-{i}.mp4'
        output_dir = f'CutFrame_Output/output{i}'
        split_video_into_parts(video_path, output_dir, frames)
def detect_frames(number_of_output, frames):
    for i in range(number_of_output):
        for j in range(frames):
            path = f'CutFrame_Output/output{i}/frame_{j}.png'
            print(path)
            output = f'CutFrame_Output/output{i}/tem.txt'
            outputpath = f'CutFrame_Output/output{i}/use{j}.txt'
            detect(path, output, outputpath)
    data =get_data('CutFrame_Output')
detect_frames(149, 7)



