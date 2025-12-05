import os

file_path = "ldm/models/diffusion/ddpm.py"

with open(file_path, "r") as f:
    content = f.read()

# 定義錯誤的與正確的迴圈寫法
wrong_loop = "for name, param in self.model.named_parameters():"
correct_loop = "for name, param in self.named_parameters():"

# 檢查是否已經有修補程式碼
if "FIX: Auto-reshape" in content:
    print("🔍 發現已存在的修補程式碼，正在檢查是否需要修正...")
    if wrong_loop in content:
        # 修正錯誤的迴圈寫法
        new_content = content.replace(wrong_loop, correct_loop)
        with open(file_path, "w") as f:
            f.write(new_content)
        print("✅ 已將 'self.model.named_parameters()' 修正為 'self.named_parameters()'！")
    elif correct_loop in content:
        print("✅ 程式碼已經是正確版本，無需修改。")
    else:
        print("⚠️ 檢測到修補區塊但格式未預期，未進行修改。")
else:
    print("🔍 未發現修補程式碼，正在執行全新植入...")
    # 如果還沒修補過，插入完整的正確代碼
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    target_line_content = "if self.make_it_fit:"
    insert_index = -1
    
    for i, line in enumerate(lines):
        if target_line_content in line:
            insert_index = i
            break
            
    if insert_index != -1:
        indent = "        " # 8個空白
        patch_code = [
            f"{indent}# FIX: Auto-reshape 2D weights to 4D for SD 2.1 compatibility\n",
            f"{indent}for name, param in self.named_parameters():\n", # 這裡是正確的寫法
            f"{indent}    if name in sd:\n",
            f"{indent}        if len(param.shape) == 4 and len(sd[name].shape) == 2:\n",
            f"{indent}            if param.shape[:2] == sd[name].shape:\n",
            f"{indent}                print(f\"Auto-reshaping {{name}} from {{sd[name].shape}} to {{param.shape}}\")\n",
            f"{indent}                sd[name] = sd[name].unsqueeze(-1).unsqueeze(-1)\n",
            f"\n"
        ]
        new_lines = lines[:insert_index] + patch_code + lines[insert_index:]
        with open(file_path, "w") as f:
            f.writelines(new_lines)
        print("✅ 成功植入自動形狀調整代碼！")
    else:
        print("❌ 找不到插入點 'if self.make_it_fit:'，請檢查檔案內容。")

