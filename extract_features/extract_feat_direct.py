import argparse
import json
import math
import os
import sys

import jsonlines
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm import tqdm

import clip

BICUBIC = InterpolationMode.BICUBIC


def clip_transform(n_px):
    return Compose(
        [
            Resize((n_px, n_px), interpolation=BICUBIC),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def time_str_to_seconds(time_str):
    if isinstance(time_str, (int, float)):
        return float(time_str)
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    except:
        return 0.0


def load_annotations(anno_path):
    annotations = {}
    if not anno_path or not os.path.exists(anno_path):
        return annotations

    with jsonlines.open(anno_path) as reader:
        for line in reader:
            video_id = line.get("YoutubeID") or line.get("video_id")
            if not video_id:
                continue

            if video_id not in annotations:
                annotations[video_id] = []

            start = time_str_to_seconds(line.get("Start_timestamp", 0))
            end = time_str_to_seconds(line.get("End_timestamp", 0))
            caption = line.get("caption", "")
            annotations[video_id].append(
                {"start": start, "end": end, "caption": caption}
            )
    print(annotations)

    return annotations


def get_frame_indices(vlen, input_fps, output_fps=4.0):
    duration = float(vlen) / input_fps
    delta = 1 / output_fps
    frame_seconds = np.arange(0 + delta / 2, duration, delta)
    frame_indices = np.around(frame_seconds * input_fps).astype(int)
    frame_indices = [e for e in frame_indices if e < vlen]
    return frame_indices


def process_video(
    video_path, output_dir, annotations, model, preprocess, device, fps=4.0
):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_id}.pth")

    if os.path.exists(output_path):
        return

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return

    vlen = len(vr)
    video_fps = vr.get_avg_fps()
    if video_fps <= 0:
        video_fps = 24.0

    frame_indices = get_frame_indices(vlen, video_fps, output_fps=fps)
    if not frame_indices:
        return

    # Extract features in batches to save GPU memory
    batch_size = 32
    all_features = []

    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i : i + batch_size]
        frames = vr.get_batch(batch_indices).asnumpy()

        batch_tensors = []
        for frame in frames:
            img = Image.fromarray(frame)
            batch_tensors.append(preprocess(img))

        input_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            batch_feat = model.encode_image(input_tensor).float().cpu()
        all_features.append(batch_feat)

    visual_features = torch.cat(all_features, dim=0)

    # Process annotations
    vlen_feat = len(frame_indices)
    video_annos = annotations.get(video_id, [])
    anno_tensor = torch.zeros(vlen_feat, dtype=torch.long)
    captions_list = []

    for i, entry in enumerate(video_annos):
        cls_id = i + 1
        start_frame = int(entry["start"] * fps)
        end_frame = math.ceil(entry["end"] * fps)

        start_frame = max(0, min(vlen_feat - 1, start_frame))
        end_frame = max(0, min(vlen_feat, end_frame))

        anno_tensor[start_frame:end_frame] = cls_id
        captions_list.append({cls_id: [entry["start"], entry["end"], entry["caption"]]})

    sample_features = {
        "rgb": visual_features,
        "anno": anno_tensor,
        "captions": captions_list,
        "fps": fps,
        "vlen": vlen_feat,
        "vlen_src": vlen,
        "src_fps": video_fps,
    }

    torch.save(sample_features, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Directly extract CLIP features from videos."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save .pth features"
    )
    parser.add_argument("--anno_path", type=str, help="Path to .jsonl annotation file")
    parser.add_argument("--model", type=str, default="ViT-B/16", help="CLIP model name")
    parser.add_argument(
        "--fps", type=float, default=4.0, help="Target FPS for feature extraction"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _ = clip.load(args.model, device=device)
    model.eval()

    preprocess = clip_transform(224)
    annotations = load_annotations(args.anno_path)

    video_extensions = (".mp4", ".mkv", ".avi", ".mov")
    videos = sorted(
        [f for f in os.listdir(args.input_dir) if f.lower().endswith(video_extensions)]
    )

    for video_name in tqdm(videos, desc="Extracting features"):
        video_path = os.path.join(args.input_dir, video_name)
        process_video(
            video_path,
            args.output_dir,
            annotations,
            model,
            preprocess,
            device,
            args.fps,
        )
