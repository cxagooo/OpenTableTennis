from moviepy.editor import VideoFileClip
from moviepy.tools import cvsecs
from PIL import Image
import os
from utils import get_data
def extract_frames(input_path, output_dir, frame_output):
    # 创建输出目录如果不存在

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载视频文件

    clip = VideoFileClip(input_path)

    # 获取视频的总时长

    duration = clip.duration

    # 计算每张图片之间的间隔

    interval = duration / frame_output - 1  # 因为我们要包括首尾帧，所以是6个间隔

    # 提取帧并保存为图片

    for i in range(frame_output):
        time_point = i * interval

        frame = clip.get_frame(time_point)

        # 保存图片到文件

        output_path = os.path.join(output_dir, f"frame_{i}.png")



        im = Image.fromarray(frame)

        im.save(output_path)

    # 释放资源

    clip.close()



#input_video_path = "path/to/your/video.mp4"

#output_directory = "path/to/output/directory"

#extract_frames(input_video_path, output_directory)