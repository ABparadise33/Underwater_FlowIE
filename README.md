# FlowIE for Underwater Image Enhancement

此專案基於 **FlowIE (CVPR 2024)** 進行修改，專門針對 **水下影像修復 (Underwater Image Enhancement)** 任務進行優化與訓練。

## 🚀 主要修改內容

本專案針對原始程式碼進行了以下改進：

* **🔧 環境適配**：修復了在 **PyTorch 2.0+** 與新版 **Diffusers** 環境下的相容性問題。
* **💾 記憶體優化**：調整訓練配置以支援 **24GB VRAM** (Consumer GPU) 進行訓練。
* **🎨 色彩修正**：移除原版不適合水下任務的 Wavelet Color Fix，改讓模型直接學習正確色調。
* **📂 資料集適配**：新增針對 **UIEB 資料集** 的讀取與評估邏輯。

---

## 🛠️ 1. 安裝 (Installation)

建議使用 Conda 建立虛擬環境 (Python 3.9+)。

### 步驟 1：建立環境與下載專案

```bash
# 1. 建立並啟動虛擬環境
conda create -n FlowIE python=3.9
conda activate FlowIE

# 2. Clone 本專案
git clone [https://github.com/ABparadise33/Underwater_FlowIE.git](https://github.com/ABparadise33/Underwater_FlowIE.git)
cd Underwater_FlowIE
# 3. 安裝 Python 依賴套件 (包含 rawpy, accelerate, pyiqa 等)
pip install -r requirements.txt

# 4. 安裝自定義 CUDA 擴充 (必須手動編譯安裝)
cd utils/torchinterp1d
pip install .
cd ../..
# 5. [重要] 安裝 lpips-pytorch (需手動修正 setup.py 版本號錯誤)
# 原作者 setup.py 寫了 version='latest' 會導致 pip 報錯，需手動改為 1.0.0
git clone [https://github.com/S-aiueo32/lpips-pytorch.git](https://github.com/S-aiueo32/lpips-pytorch.git)
sed -i "s/version='latest'/version='1.0.0'/" lpips-pytorch/setup.py
pip install ./lpips-pytorch
rm -rf lpips-pytorch

# 6. [重要] 修復 Basicsr 與 Torchvision 版本不相容問題
# 這行指令會自動修正虛擬環境中 basicsr 的錯誤引用
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' $(python -c "import basicsr; import os; print(os.path.dirname(basicsr.__file__))")/data/degradations.py
# 1. 建立資料夾
mkdir -p datasets
cd datasets

# 2. 下載資料集 (需安裝 git-lfs)
# 若未安裝 git-lfs，請先執行: apt-get update && apt-get install git-lfs -y && git lfs install
git clone [https://huggingface.co/datasets/Edddddd8787/temp-weights](https://huggingface.co/datasets/Edddddd8787/temp-weights) underwater_train

# 3. 整理資料夾結構 (改名 + 刪除 .git)
cd underwater_train
rm -rf .git
mv raw-890 underwater
mv reference-890 GT
cd ../..

# 4. 自動切分訓練集與測試集 (800 Train / 90 Test)
# 此腳本會將部分圖片移動到 datasets/underwater_test
python split_dataset.py
datasets/
├── underwater_train/  (800 pairs for training)
│   ├── GT/            (Ground Truth)
│   └── underwater/    (Input Images)
└── underwater_test/   (90 pairs for evaluation)
    ├── GT/
    └── underwater/
# 1. 確保 weights 資料夾存在
mkdir -p weights

# 2. 下載 Stable Diffusion v2.1 Base (~5.2GB)
wget -O weights/v2-1_512-ema-pruned.ckpt [https://huggingface.co/camenduru/unianimate/resolve/main/v2-1_512-ema-pruned.ckpt](https://huggingface.co/camenduru/unianimate/resolve/main/v2-1_512-ema-pruned.ckpt)

# 3. 下載 SwinIR Initial Module (~60MB)
wget -O weights/general_swinir_v1.ckpt [https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt](https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt)
python train.py --config ./configs/train_cldm_underwater.yaml
# 請將 CKPT_PATH 替換為訓練出來的權重檔 (例如 lightning_logs/version_0/checkpoints/last.ckpt)
python inference_bsr.py \
  --ckpt CKPT_PATH \
  --input ./datasets/underwater_test/underwater \
  --output ./results/underwater_inference \
  --sr_scale 1 \
  --tiled
python evaluate.py \
  --input1 ./results/underwater_inference \
  --input2 ./datasets/underwater_test/GT \
  --type all
