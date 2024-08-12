from moviepy.editor import VideoFileClip, vfx
from PIL import Image
import os


def adjust_and_extract_frames(input_path, output_dir, num_frames, target_duration=1.4):
    # 创建输出目录如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载视频文件
    clip = VideoFileClip(input_path)

    # 获取视频的原始时长
    duration = clip.duration


    # 计算每个帧之间的时间间隔
    if num_frames > 1:
        interval = duration / (num_frames - 1)
    else:
        interval = 0

    # 最小时间间隔，例如1/60秒
    min_interval = 1 / 60

    # 生成时间戳列表
    timestamps = [i * interval for i in range(num_frames)]

    # 确保时间戳不会过于接近
    for i in range(len(timestamps) - 1):
        if timestamps[i + 1] - timestamps[i] < min_interval:
            timestamps[i + 1] = timestamps[i] + min_interval

    # 添加最后一个时间戳，确保它正好等于视频的总时长
    if num_frames > 1:
        timestamps[-1] = duration

    # 提取帧并保存为图片
    for i, timestamp in enumerate(timestamps):
        frame = clip.get_frame(timestamp)

        # 保存图片到文件
        output_path = os.path.join(output_dir, f"frame_{i}.png")
        im = Image.fromarray(frame)
        im.save(output_path)

    # 删除最后一个帧的照片
    last_frame_path = os.path.join(output_dir, f"frame_{num_frames - 1}.png")
    if os.path.exists(last_frame_path):
        os.remove(last_frame_path)

    # 释放资源
    clip.close()










