import os
import shutil
import random

# 設定路徑
base_dir = "./datasets"
train_dir = os.path.join(base_dir, "underwater_train")
test_dir = os.path.join(base_dir, "underwater_test")

# 來源資料夾
src_input = os.path.join(train_dir, "underwater")
src_gt = os.path.join(train_dir, "GT")

# 目標資料夾 (測試集)
dst_input = os.path.join(test_dir, "underwater")
dst_gt = os.path.join(test_dir, "GT")

# 確保來源存在
if not os.path.exists(src_input) or not os.path.exists(src_gt):
    print("❌ 錯誤：找不到來源資料夾 (datasets/underwater_train)")
    exit()

# 確保目標資料夾存在
os.makedirs(dst_input, exist_ok=True)
os.makedirs(dst_gt, exist_ok=True)

# 取得所有圖片列表 (只看 GT 資料夾，確保成對)
all_files = sorted([f for f in os.listdir(src_gt) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
total_count = len(all_files)

print(f"📂 總共有 {total_count} 張圖片。")

if total_count <= 90:
    print("⚠️ 圖片數量太少，無法切分！")
    exit()

# 設定切分數量
test_count = 90
train_count = total_count - test_count

print(f"🔪 準備切分：訓練集 {train_count} 張 / 測試集 {test_count} 張")

# 隨機抽樣
random.seed(42) # 固定種子，確保每次切分結果一樣
test_files = random.sample(all_files, test_count)

# 開始移動
count = 0
for filename in test_files:
    # 建構完整路徑
    s_gt = os.path.join(src_gt, filename)
    d_gt = os.path.join(dst_gt, filename)
    
    # 嘗試對應 input (有時候副檔名不同，這裡假設檔名一致)
    # 如果您的 input 是 raw-890 來的，檔名應該跟 GT 一樣
    s_in = os.path.join(src_input, filename)
    d_in = os.path.join(dst_input, filename)
    
    if os.path.exists(s_gt) and os.path.exists(s_in):
        shutil.move(s_gt, d_gt)
        shutil.move(s_in, d_in)
        count += 1
    else:
        print(f"⚠️ 找不到配對檔案，跳過: {filename}")

print(f"✅ 完成！已移動 {count} 對圖片到 {test_dir}")
print(f"   訓練集剩餘: {len(os.listdir(src_gt))} 張")
print(f"   測試集現有: {len(os.listdir(dst_gt))} 張")
