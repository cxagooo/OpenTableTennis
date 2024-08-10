from demo import detect
from OpenAndPick import pick
from cut import extract_frames
import os
#截取帧
for i in range(24):
    #os.mkdir(f'CutFrame_Output/output{i}')
    video_path = f'/home/cxgao/Videos/pyopen/12-标记-{i}.mp4'
    output_dir = f'CutFrame_Output/output{i}'
    extract_frames(video_path, output_dir)
for i in range(24):
    for j in range(3):
        path = f'CutFrame_Output/output{i}/frame_{j}.jpg'
        print(path)
        output = f'CutFrame_Output/output{i}/tem.txt'
        outputpath = f'CutFrame_Output/output{i}/use{j}.txt'
        detect(path, output, outputpath)



