import os

file_path = "ldm/models/diffusion/ddpm.py"

with open(file_path, "r") as f:
    lines = f.readlines()

# 我們要插入修復邏輯的位置：在 'if self.make_it_fit:' 之前
# 這樣無論是否開啟 make_it_fit，都會先修復這個已知的維度問題
target_line_content = "if self.make_it_fit:"
insert_index = -1

for i, line in enumerate(lines):
    if target_line_content in line:
        insert_index = i
        break

if insert_index != -1:
    # 定義要插入的修復程式碼 (注意縮排要對齊)
    indent = "        " # 8個空白
    patch_code = [
        f"{indent}# FIX: Auto-reshape 2D weights to 4D for SD 2.1 compatibility\n",
        f"{indent}for name, param in self.model.named_parameters():\n",
        f"{indent}    if name in sd:\n",
        f"{indent}        if len(param.shape) == 4 and len(sd[name].shape) == 2:\n",
        f"{indent}            if param.shape[:2] == sd[name].shape:\n",
        f"{indent}                print(f\"Auto-reshaping {{name}} from {{sd[name].shape}} to {{param.shape}}\")\n",
        f"{indent}                sd[name] = sd[name].unsqueeze(-1).unsqueeze(-1)\n",
        f"\n"
    ]
    
    # 插入程式碼
    new_lines = lines[:insert_index] + patch_code + lines[insert_index:]
    
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("✅ Successfully patched ddpm.py for SD 2.1 dimensions mismatch!")
else:
    print("❌ Could not find insertion point in ddpm.py")
