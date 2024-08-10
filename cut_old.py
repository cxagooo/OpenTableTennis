import cv2
import os

def extract_frames(video_path, output_dir):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频的总帧数
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算中间帧的位置
    #mid_frame_index = frame_count // 2

    # 初始化帧计数器
    frame_counter = 0
    n = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # 检查当前帧是否为所需帧
        if frame_counter in (0, mid_frame_index, frame_count - 1):
            # 构建输出文件名
            filename = f"frame_{n}.jpg"
            filepath = output_dir + '/' + filename
            n += 1
            # 保存帧到文件
            cv2.imwrite(filepath, frame)
        frame_counter += 1

    # 释放资源
    video.release()


# 设置视频路径和输出目录
#video_path = '/home/cxgao/Videos/pyopen/12-标记-0.mp4'
#output_dir = 'CutFrame_Output'

# 调用函数
#extract_frames(video_path, output_dir)
'''for i in range(24):
    os.mkdir(f'CutFrame_Output/output{i}')
    video_path = f'/home/cxgao/Videos/pyopen/12-标记-{i}.mp4'
    output_dir = f'CutFrame_Output/output{i}'
    extract_frames(video_path, output_dir)'''