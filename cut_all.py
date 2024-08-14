import cv2
import os

def extract_frames(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 检查是否成功打开
    if not video.isOpened():
        print("Error opening video file")
        return

    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        # 保存帧到文件
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # 释放资源
    video.release()

    print(f"Extracted {frame_count} frames to {output_folder}")

# 使用示例
if __name__ == '__main__':
    for i in range(149):
        video_path = f'/home/cxgao/Videos/data3/31-标记-{i}.mp4'
        output_dir = f'AllFrame_Output/output{i}'
        extract_frames(video_path, output_dir)