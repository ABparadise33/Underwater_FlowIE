## FlowIE for Underwater Image Enhancement

此專案基於 **[FlowIE (CVPR 2024)](https://github.com/EternalEvan/FlowIE)** 進行修改，專門針對 **水下影像修復 (Underwater Image Enhancement)** 任務進行優化與訓練。

主要修改包含：
* **環境適配**：修復了在 PyTorch 2.0+ 與新版 Diffusers 環境下的相容性問題。
* **記憶體優化**：調整訓練配置以支援 24GB VRAM (Consumer GPU) 訓練。
* **色彩修正**：移除原版不適合水下任務的 Wavelet Color Fix，改讓模型直接學習正確色調。
* **資料集適配**：新增針對 UIEB 資料集的讀取與評估邏輯。

---

## 🛠️ 1. 安裝 (Installation)

建議使用 Conda 建立虛擬環境 (Python 3.9+)。

```bash
# 1. 建立並啟動虛擬環境
conda create -n FlowIE python=3.9
conda activate FlowIE

# 2. Clone 本專案
git clone https://github.com/ABparadise33/Underwater_FlowIE.git
cd Underwater_FlowIE

# 3. 安裝 Python 依賴套件 (包含 rawpy, accelerate, pyiqa 等所有依賴)
pip install -r requirements.txt

# 4. 安裝自定義 CUDA 擴充 (必須手動編譯安裝)
cd utils/torchinterp1d
pip install .
cd ../..

# 5. [重要] 安裝 lpips-pytorch (需手動修正版本號錯誤)
# 步驟 3 的 requirements.txt 內建安裝。這裡處理錯誤版本號問題
git clone https://github.com/S-aiueo32/lpips-pytorch.git
# 使用 sed 修正 setup.py 內的錯誤版本號
sed -i "s/version='latest'/version='1.0.0'/" lpips-pytorch/setup.py 
pip install ./lpips-pytorch
rm -rf lpips-pytorch

# 6. [重要] 修復 Basicsr 與 Torchvision 版本不相容問題
# 這行指令會自動修正虛擬環境中 basicsr 的錯誤引用
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' $(python -c "import basicsr; import os; print(os.path.dirname(basicsr.__file__))")/data/degradations.py

## 2. 資料集準備 (Data Preparation)
本專案使用 UIEB (Underwater Image Enhancement Benchmark) 資料集。

請執行以下指令，自動從 Hugging Face 下載並切分資料集：
# 1. 建立資料夾
mkdir -p datasets
cd datasets

# 2. 下載資料集 (需安裝 git-lfs)
# 若未安裝: apt-get update && apt-get install git-lfs -y && git lfs install
git clone [https://huggingface.co/datasets/Edddddd8787/temp-weights](https://huggingface.co/datasets/Edddddd8787/temp-weights) underwater_train

# 3. 整理資料夾結構 (改名 + 刪除 .git)
cd underwater_train
rm -rf .git
mv raw-890 underwater
mv reference-890 GT
cd ../..

# 4. 自動切分訓練集與測試集 (800 Train / 90 Test)
python split_dataset.py

完成後的結構應如下：
datasets/
├── underwater_train/ (800 pairs for training)
│   ├── GT/
│   └── underwater/
└── underwater_test/  (90 pairs for evaluation)
    ├── GT/
    └── underwater/

## 3. 下載預訓練權重 (Pretrained Weights)
FlowIE 需要 Stable Diffusion v2.1 作為底層模型，以及 SwinIR 作為初始特徵提取器。
# 1. 確保 weights 資料夾存在
mkdir -p weights

# 2. Stable Diffusion v2.1 Base (~5.2GB)
wget -O weights/v2-1_512-ema-pruned.ckpt [https://huggingface.co/camenduru/unianimate/resolve/main/v2-1_512-ema-pruned.ckpt](https://huggingface.co/camenduru/unianimate/resolve/main/v2-1_512-ema-pruned.ckpt)

# 3. SwinIR Initial Module (~60MB)
wget -O weights/general_swinir_v1.ckpt [https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt](https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt)

## 4. 訓練 (Training)
執行以下指令開始訓練。設定檔已針對 24GB VRAM 優化（Batch Size=1, Gradient Accumulation=4）。=
python train.py --config ./configs/train_cldm_underwater.yaml
