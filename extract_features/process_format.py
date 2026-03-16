import pathlib

import pandas as pd
import torch
from tqdm import tqdm

# 配置路径
src_path = pathlib.Path("/home/thw/project/OV-OAD/data/Thumos14D")
save_path = pathlib.Path("/home/thw/project/OV-OAD/data/Thumos14D_Processed")
save_path.mkdir(exist_ok=True)


def process_thumos():
    # 1. 预扫描所有文件，收集唯一的动作类别，建立全局映射
    all_categories = set()
    print("正在扫描动作类别...")
    for p in src_path.glob("*.pth"):
        data = torch.load(p)
        for item in data["captions"]:
            # item 格式如: {1: [8.6, 12.1, 'CricketBowling']}
            for key in item:
                all_categories.add(item[key][2])

    # 建立类别到全局 ID 的映射 (背景设为 0，动作从 1 开始)
    sorted_categories = sorted(list(all_categories))
    label_to_id = {name: i + 1 for i, name in enumerate(sorted_categories)}
    label_to_id["Background"] = 0

    # 保存 meta.csv
    meta_data = [
        {"action_name": name, "label_id": idx} for name, idx in label_to_id.items()
    ]
    df_meta = pd.DataFrame(meta_data)
    df_meta.to_csv(save_path / "meta.csv", index=False)
    print(f"已生成 meta.csv，共包含 {len(label_to_id)} 个类别（含背景）")

    # 2. 转换数据结构并保存
    print("正在转换数据结构...")
    for p in tqdm(list(src_path.glob("*.pth"))):
        data = torch.load(p)

        # 构建当前视频的 局部ID -> 全局ID 映射表
        # local_to_global 映射原 anno 中的数字到新 label_id
        local_to_global = {0: 0}  # 背景始终保持为 0
        for item in data["captions"]:
            for local_id, info in item.items():
                category_name = info[2]
                local_to_global[local_id] = label_to_id[category_name]

        # 转换 anno 张量
        # 使用 map 函数或直接对张量元素进行替换
        old_anno = data["anno"].tolist()
        new_anno = torch.tensor(
            [local_to_global[val] for val in old_anno], dtype=torch.long
        )

        # 构建新的字典
        new_data = {
            "rgb": data["rgb"],
            "anno": new_anno,
            "fps": data["fps"],
            "vlen": data["vlen"],
            "vlen_src": data.get("vlen_src"),
            "src_fps": data.get("src_fps"),
        }

        # 保存新文件
        torch.save(new_data, save_path / p.name)


if __name__ == "__main__":
    process_thumos()
