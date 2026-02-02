import os
import cv2
from tqdm import tqdm

def save_video_frames(input_dir, output_dir, taregt_resolution=[512, 320]):

    os.makedirs(output_dir, exist_ok=True)
    
    dir_frame_idx = None
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(input_dir, filename)
            
            base_name = os.path.splitext(filename)[0]
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open {video_path}")
                continue

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                output_filename = f"{base_name}_frame_{frame_idx}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                if taregt_resolution is not None:
                    # 调整帧尺寸
                    target_w, target_h = taregt_resolution
                    resized_frame = cv2.resize(frame, (target_w, target_h), cv2.INTER_LINEAR)
                else:
                    resized_frame = frame

                if not cv2.imwrite(output_path, resized_frame):
                    print(f"Warnning: Cannot write {output_path}")
                
                frame_idx += 1

            cap.release()
            
            if dir_frame_idx == None:
                dir_frame_idx = frame_idx
            else:
                if dir_frame_idx != frame_idx:
                    print(f"Warning: {video_path} has {frame_idx} frames, but {dir_frame_idx} frames in {input_dir}")
                dir_frame_idx = min(dir_frame_idx, frame_idx)
                
    return dir_frame_idx

if __name__ == "__main__":
    save_video_frames(
        input_dir="aaa",
        output_dir="bbb"
    )