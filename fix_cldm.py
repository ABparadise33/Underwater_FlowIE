import os

file_path = "model/cldm.py"
target_line = "self.preprocess_model = instantiate_from_config(preprocess_config)"

with open(file_path, "r") as f:
    lines = f.readlines()

found = False
for i, line in enumerate(lines):
    if target_line in line:
        # 取得縮排 (Indentation)
        indent = line[:line.find(target_line)]
        
        # 準備要插入的修正程式碼
        new_block = [
            f"{indent}# FIX: Extract ckpt_path and load manually to avoid __init__ error\n",
            f"{indent}swinir_ckpt = None\n",
            f"{indent}if 'params' in preprocess_config and 'ckpt_path' in preprocess_config['params']:\n",
            f"{indent}    swinir_ckpt = preprocess_config['params'].pop('ckpt_path')\n",
            f"{indent}\n",
            f"{indent}# Original instantiation\n",
            f"{indent}{target_line}\n",
            f"{indent}\n",
            f"{indent}# Load weights if path provided\n",
            f"{indent}if swinir_ckpt is not None:\n",
            f"{indent}    print(f'Loading SwinIR weights from {{swinir_ckpt}}')\n",
            f"{indent}    try:\n",
            f"{indent}        # Fix for PyTorch 2.6+ security check\n",
            f"{indent}        sd = torch.load(swinir_ckpt, map_location='cpu', weights_only=False)\n",
            f"{indent}    except TypeError:\n",
            f"{indent}        sd = torch.load(swinir_ckpt, map_location='cpu')\n",
            f"{indent}    if 'state_dict' in sd:\n",
            f"{indent}        sd = sd['state_dict']\n",
            f"{indent}    self.preprocess_model.load_state_dict(sd, strict=False)\n"
        ]
        
        lines[i] = "".join(new_block)
        found = True
        break

if found:
    with open(file_path, "w") as f:
        f.writelines(lines)
    print("✅ Successfully patched model/cldm.py to handle SwinIR weights!")
else:
    print("❌ Could not find target line in model/cldm.py. Has it been modified?")
