import numpy as np
import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import pyiqa

def main(path1, path2, type="all"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 初始化 Metrics
    metrics = {}
    if type in ["psnr", "all"]:
        metrics["psnr"] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
    if type in ["ssim", "all"]:
        metrics["ssim"] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
    if type in ["lpips", "all"]:
        metrics["lpips"] = pyiqa.create_metric('lpips', device=device)
    if type in ["musiq", "all"]:
        metrics["musiq"] = pyiqa.create_metric('musiq', device=device)

    # 讀取 Restored (Output) 資料夾的檔案
    files1 = sorted([f for f in os.listdir(path1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # 建立 GT 資料夾的索引 (檔名不含副檔名 -> 完整路徑)
    # 這樣可以忽略 .jpg 和 .png 的差異
    gt_map = {}
    for f in os.listdir(path2):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            name_no_ext = os.path.splitext(f)[0]
            gt_map[name_no_ext] = os.path.join(path2, f)

    print(f"Restored folder: {len(files1)} images")
    print(f"GT folder:       {len(gt_map)} images")

    scores = {k: [] for k in metrics.keys()}
    paired_count = 0
    unpaired_files = []

    print("\n[Pairing Check] First 5 matches:")

    for i, img_name in enumerate(tqdm(files1, desc="Evaluating")):
        imgpath1 = os.path.join(path1, img_name)
        
        # --- 關鍵修正邏輯 ---
        # 取得檔名 (不含路徑和副檔名)
        base_name = os.path.splitext(img_name)[0]
        
        # 如果檔名結尾是 '_0'，就把它切掉！
        # 例如: '85_img__0' -> '85_img_'
        if base_name.endswith('_0'):
            real_name = base_name[:-2]
        else:
            real_name = base_name
            
        # 去 GT map 找有沒有這個名字
        imgpath2 = gt_map.get(real_name)
        
        has_gt = imgpath2 is not None

        # 顯示前幾個配對結果給您檢查
        if i < 5:
            match_str = os.path.basename(imgpath2) if has_gt else "NO MATCH"
            tqdm.write(f"  {img_name} -> {match_str}")

        if not has_gt:
            if type != "musiq":
                unpaired_files.append(img_name)
                continue
        else:
            paired_count += 1

        # 計算指標
        with torch.no_grad():
            for name, metric in metrics.items():
                if name == "musiq":
                    val = metric(imgpath1).item()
                    scores[name].append(val)
                elif has_gt:
                    val = metric(imgpath1, imgpath2).item()
                    scores[name].append(val)

    # --- 輸出結果 ---
    print("\n" + "="*30)
    print("       EVALUATION REPORT       ")
    print("="*30)
    print(f"Total Restored Images: {len(files1)}")
    print(f"Successfully Paired:   {paired_count}")
    
    if unpaired_files:
        print(f"Unpaired Images:       {len(unpaired_files)}")
        print("Example unpaired files (could not find GT):")
        print(unpaired_files[:5])

    print("-" * 30)
    for name, vals in scores.items():
        if vals:
            avg = np.mean(vals)
            arrow = "↓" if name == "lpips" else "↑"
            print(f"{name.upper():<6} {arrow} : {avg:.4f}")
        else:
            print(f"{name.upper():<6}   : No data")
    print("="*30)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input1", type=str, required=True, help="Path to restored images")
    parser.add_argument("--input2", type=str, required=True, help="Path to GT images")
    parser.add_argument("--type", type=str, default="all", choices=["psnr", "ssim", "lpips", "musiq", "all"])
    args = parser.parse_args()
    
    if not os.path.exists(args.input1):
        print(f"Error: Input path {args.input1} does not exist.")
    else:
        main(args.input1, args.input2, args.type)