import os

# --- 任務 1: 修改 ImageLogger 讓資料夾整齊 ---
callback_path = "model/callbacks.py"
with open(callback_path, "r") as f:
    content = f.read()

# 原始的儲存邏輯 (全部混在一起)
old_save_logic = """                filename = "{}_step-{:06}_e-{:06}_b-{:06}.png".format(
                    image_key, pl_module.global_step, pl_module.current_epoch, batch_idx
                )
                path = os.path.join(save_dir, filename)
                Image.fromarray(grid).save(path)"""

# 新的儲存邏輯 (分資料夾)
new_save_logic = """                # Modified: Save to subfolders by type
                type_dir = os.path.join(save_dir, image_key)
                os.makedirs(type_dir, exist_ok=True)
                
                filename = "step-{:06}_e-{:06}_b-{:06}.png".format(
                    pl_module.global_step, pl_module.current_epoch, batch_idx
                )
                path = os.path.join(type_dir, filename)
                Image.fromarray(grid).save(path)"""

if old_save_logic in content:
    content = content.replace(old_save_logic, new_save_logic)
    with open(callback_path, "w") as f:
        f.write(content)
    print("✅ 已優化 model/callbacks.py：圖片將自動分類到子資料夾！")
else:
    if "type_dir =" in content:
        print("ℹ️ model/callbacks.py 已經優化過了，無需修改。")
    else:
        print("⚠️ 無法定位 model/callbacks.py 中的儲存邏輯，請檢查檔案內容。")

# --- 任務 2: 修改 Color Fix Type 為 None ---
cldm_path = "model/cldm.py"
with open(cldm_path, "r") as f:
    cldm_content = f.read()

if 'color_fix_type="wavelet"' in cldm_content:
    cldm_content = cldm_content.replace('color_fix_type="wavelet"', 'color_fix_type="none"')
    with open(cldm_path, "w") as f:
        f.write(cldm_content)
    print("✅ 已優化 model/cldm.py：將 color_fix_type 設為 'none' (適合水下修復)。")
else:
    print("ℹ️ model/cldm.py 已經是 none 或已被修改過。")

