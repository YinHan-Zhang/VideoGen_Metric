import os
import re
import json

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from utils import (
    calculate_psnr,
    calculate_ssim,
    calculate_fvd,
    calculate_epe,
    calculate_lpips,
    calculate_fid,
    calculate_clip_I,
    save_video_frames,
    preprocess
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_in_chunks(all_raw_videos, all_gen_videos, batch_size, target_resolution=(224, 224)):

    processed_raw_chunks = []
    processed_gen_chunks = []

    for i in range(0, len(all_raw_videos), batch_size):
        raw_chunk_videos = torch.cat(all_raw_videos[i:i + batch_size], dim=0)  # (batch_size * T, C, H, W)
        gen_chunk_videos = torch.cat(all_gen_videos[i:i + batch_size], dim=0)
        
        raw_chunk_processed = preprocess(raw_chunk_videos, target_resolution)  # 返回 (batch_size, C, T, H', W')
        gen_chunk_processed = preprocess(gen_chunk_videos, target_resolution)  # 同上

        processed_raw_chunks.append(raw_chunk_processed)
        processed_gen_chunks.append(gen_chunk_processed)

    processed_raw = torch.cat(processed_raw_chunks, dim=0)
    processed_gen = torch.cat(processed_gen_chunks, dim=0)

    return processed_raw, processed_gen

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_min_max_frame(frames_dir):
    frame_pattern = re.compile(r'^(.*?)_frame_(\d+)\.png$')
    max_frames = {}

    for filename in os.listdir(frames_dir):
        if not filename.endswith('.png'):
            continue
        match = frame_pattern.match(filename)
        if not match:
            continue
        video_name, frame_num = match.groups()
        frame_num = int(frame_num)
        current_max = max_frames.get(video_name, -1)
        if frame_num > current_max:
            max_frames[video_name] = frame_num

    return min(max_frames.values()) if max_frames else 0

def main():
    
    raw_root = ""
    gen_root = ""

    # taregt_resolution = None
    taregt_resolution = [512, 320] # 如果生成的视频的尺寸与原视频不一样，以生成的视频尺寸为主

    raw_frame_dir = f"{raw_root}_frames"
    gen_frame_dir = f"{gen_root}_frames"
    
    if not os.path.exists(raw_frame_dir):
        raw_frame_num = save_video_frames(raw_root, raw_frame_dir, taregt_resolution)
    else:
        raw_frame_num = get_min_max_frame(raw_frame_dir)

    if not os.path.exists(gen_frame_dir):
        gen_frame_num = save_video_frames(gen_root, gen_frame_dir, taregt_resolution)
    else:
        gen_frame_num = get_min_max_frame(gen_frame_dir)
        
    print(f"Evaluating with frame count: {gen_frame_num}")
    # assert gen_frame_num <= raw_frame_num, "Generated frames exceed raw frames count"
    
    video_names = sorted([name for name in os.listdir(gen_root) if name.endswith('.mp4')])
    
    scores = {
        "clip": [],
        "epe": [],
        "lpips": [],
        "ssim": [],
        "psnr": [],
    }
    all_raw_videos, all_gen_videos = [], []

    with torch.no_grad():
        progress_bar = tqdm(video_names, desc="Processing videos")

        for video_name in progress_bar:
            base_name = video_name.replace(".mp4", "")
            clip, lpips, ssim, psnr = [], [], [], []
            raw_video, gen_video = [], []

            # # for frame_idx in range(gen_frame_num):
            for frame_idx in range(16):
                raw_path = f"{raw_frame_dir}/{base_name}_frame_{frame_idx}.png"
                gen_path = f"{gen_frame_dir}/{base_name}_frame_{frame_idx}.png"

                try:
                    raw_img = Image.open(raw_path)
                    gen_img = Image.open(gen_path)
                except FileNotFoundError:
                    break


                # Align the size
                if raw_img.size != gen_img.size:
                    gen_img = gen_img.resize(raw_img.size)

                # Calculate metrics
                clip.append(calculate_clip_I(raw_img, gen_img))
                
                raw_tensor = transforms.ToTensor()(raw_img).unsqueeze(0)
                gen_tensor = transforms.ToTensor()(gen_img).unsqueeze(0)
                
                raw_video.append(raw_tensor)
                gen_video.append(gen_tensor)
                
                psnr.append(calculate_psnr(raw_tensor, gen_tensor).item())
                ssim.append(calculate_ssim(raw_tensor, gen_tensor).item())
                lpips.append(calculate_lpips(
                    raw_tensor.sub(0.5).div(0.5),
                    gen_tensor.sub(0.5).div(0.5)
                ).item())

            if not raw_video:
                continue

            # Process video-level metrics
            raw_video = torch.cat(raw_video)
            gen_video = torch.cat(gen_video)
            all_raw_videos.append(raw_video.unsqueeze(0))
            all_gen_videos.append(gen_video.unsqueeze(0))
            
            epe = calculate_epe(raw_video, gen_video).item()
            
            scores["clip"].append(np.mean(clip))
            scores["epe"].append(epe)
            scores["lpips"].append(np.mean(lpips))
            scores["ssim"].append(np.mean(ssim))
            scores["psnr"].append(np.mean(psnr))
            
            # Update progress_bar
            current_means = {
                k: round(np.mean(v), 2) 
                for k, v in scores.items() 
                if isinstance(v, list) and len(v) > 0
            }
            progress_bar.set_postfix(current_means)

        # FID
        try:
            fid = calculate_fid(raw_frame_dir, gen_frame_dir)
        except Exception as e:
            print(f"[WARN] FID calculation failed: {e}")
        else:
            scores["fid"] = fid
        
        # FVD
        processed_raw_chunks = []
        processed_gen_chunks = []

        batch_size = 20
        TARGET_RESOLUTION = (224, 224)


        for i in tqdm(range(0, len(all_raw_videos), batch_size)):

            raw_chunk_videos = torch.cat(all_raw_videos[i:i + batch_size]).mul(255).clamp(0, 255).byte().numpy()
            gen_chunk_videos = torch.cat(all_gen_videos[i:i + batch_size]).mul(255).clamp(0, 255).byte().numpy()
            raw_chunk_videos = raw_chunk_videos.transpose(0, 1, 3, 4, 2)  # [N, T, H, W, C]
            gen_chunk_videos = gen_chunk_videos.transpose(0, 1, 3, 4, 2)
            
            raw_chunk_processed = preprocess(raw_chunk_videos, TARGET_RESOLUTION)
            gen_chunk_processed = preprocess(gen_chunk_videos, TARGET_RESOLUTION)

            processed_raw_chunks.append(raw_chunk_processed)
            processed_gen_chunks.append(gen_chunk_processed)

        all_raw = torch.cat(processed_raw_chunks, dim=0)
        all_gen = torch.cat(processed_gen_chunks, dim=0)
        
        fvd = calculate_fvd(all_raw, all_gen)
        scores["fvd"] = fvd

        
        # Generate final results
        final_scores = {
            k: np.mean(v) if isinstance(v, list) else v
            for k, v in scores.items()
        }
        
        print("\nEvaluation Results:")
        for k, v in final_scores.items():
            print(f"{k.upper():<8}: {v:.4f}")
        
        results = {
            "raw_scores": scores,
            "final_scores": final_scores
        }
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        print("\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    main()