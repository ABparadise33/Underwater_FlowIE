import os
import shutil
from huggingface_hub import snapshot_download

# ================= 配置區域 =================
# 請填寫您的 Hugging Face 倉庫 ID (格式: 帳號/倉庫名)
# 例如: "User123/UIEB_Dataset"
REPO_ID = "Edddddd8787/temp-weights" 

# 設定下載目標路徑
LOCAL_DIR = "./datasets/underwater_train"
# ===========================================

print(f"🚀 開始從 Hugging Face 下載 {REPO_ID} ...")

try:
    # 下載整個倉庫 (支援斷點續傳)
    # 如果是私人倉庫，請先執行 huggingface-cli login
    path = snapshot_download(
        repo_id=REPO_ID, 
        repo_type="dataset", 
        local_dir=LOCAL_DIR, 
        resume_download=True
    )
    print("✅ 下載完成！正在整理資料夾結構...")

    # 定義來源與目標名稱
    # 您的原始資料夾名稱
    src_input = os.path.join(LOCAL_DIR, "raw-890")
    src_gt = os.path.join(LOCAL_DIR, "reference-890")

    # FlowIE 需要的資料夾名稱
    dst_input = os.path.join(LOCAL_DIR, "underwater")
    dst_gt = os.path.join(LOCAL_DIR, "GT")

    # 自動改名/移動
    if os.path.exists(src_input):
        if os.path.exists(dst_input):
            print(f"⚠️ 目標資料夾 {dst_input} 已存在，正在合併...")
            for file in os.listdir(src_input):
                shutil.move(os.path.join(src_input, file), dst_input)
            os.rmdir(src_input)
        else:
            os.rename(src_input, dst_input)
            print(f"📂 已重新命名: raw-890 -> underwater")
    
    if os.path.exists(src_gt):
        if os.path.exists(dst_gt):
            print(f"⚠️ 目標資料夾 {dst_gt} 已存在，正在合併...")
            for file in os.listdir(src_gt):
                shutil.move(os.path.join(src_gt, file), dst_gt)
            os.rmdir(src_gt)
        else:
            os.rename(src_gt, dst_gt)
            print(f"📂 已重新命名: reference-890 -> GT")

    print("\n🎉 資料集準備就緒！結構檢查：")
    print(f"   Input (underwater): {len(os.listdir(dst_input))} 張圖片")
    print(f"   Target (GT):        {len(os.listdir(dst_gt))} 張圖片")

except Exception as e:
    print(f"\n❌ 發生錯誤: {e}")
    print("提示: 如果是 404/401 錯誤，請確認 Repository ID 是否正確，或是私人倉庫是否已登入。")
EOF