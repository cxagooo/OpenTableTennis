from demo import detect
from OpenAndPick import pick
from cut import adjust_and_extract_frames
import os
from utils import get_data
#截取帧
#for i in range(24):
#    #os.mkdir(f'CutFrame_Output/output{i}')
#    video_path = f'//home/cxgao/Videos/pyopen/12-标记-{i}.mp4'
#    output_dir = f'CutFrame_Output/output{i}'
#    adjust_and_extract_frames(video_path, output_dir, 8)
for i in range(55):
    #os.mkdir(f'CutFrame_Output/output{i}')
    #p = i + 24
    video_path = f'/home/cxgao/Videos/data/22-标记-{i}.mp4'
    output_dir = f'CutFrame_Output/output{i}'
    adjust_and_extract_frames(video_path, output_dir, 4)
for i in range(55):
    for j in range(3):
        path = f'CutFrame_Output/output{i}/frame_{j}.png'
        print(path)
        output = f'CutFrame_Output/output{i}/tem.txt'
        outputpath = f'CutFrame_Output/output{i}/use{j}.txt'
        detect(path, output, outputpath)
data =get_data('CutFrame_Output')



