import os
from moviepy.editor import VideoFileClip
from PIL import Image

def split_video_into_parts(input_file, output_dir, num_parts):
    # 加载视频
    video = VideoFileClip(input_file)

    # 计算每个部分的持续时间
    part_duration = video.duration / num_parts

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分割视频并提取每部分的第一帧
    for i in range(num_parts):
        start_time = i * part_duration
        end_time = min(start_time + part_duration, video.duration)

        # 获取该部分的第一帧
        frame = video.get_frame(start_time)

        # 将帧保存为图像文件
        output_path = os.path.join(output_dir, f'frame_{i}.png')
        frame = frame.astype('uint8')
        frame = Image.fromarray(frame)
        frame.save(output_path)

    # 关闭视频文件
    video.close()


if __name__ == '__main__':
    input_video = 'path/to/your/video.mp4'
    output_directory = 'path/to/output/directory'
    number_of_parts = 5  # 你想将视频分成几部分

    split_video_into_parts(input_video, output_directory, number_of_parts)