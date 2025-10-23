# Classification Evaluation on Pidray and OPIXray

The code is built on the official STING-BEE repository, due to the lack of official requirements, please be sure to follow the steps below to configure the environment.

---

## ✅ Supported Models

| Model | Resolution |
|-------|------------|
| STING-BEE | `504×504` |
| InternVL 3.5 | `448×448` (dynamic) |
| LLaVA-1.5 | `336×336` |
| Qwen2.5-VL | `448×448` |

## 🚀 Get Started

### 1. Environment Setup

Ensure you have Python ≥ 3.9 and PyTorch installed. We recommend using a virtual environment.<br>
Note that the packages below had better be aligned or STING-BEE won't be reproducible.


```bash
git clone https://github.com/lilitao0517/Eval-Test-tfl.git
cd Eval-Test-tfl
conda create -n stingbee python=3.9
conda activate stingbee
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.31.0 #important
```

The rest of the configuration can be configured by referring directly to ms-swift or llama-factory, or by aligning the requirements.txt in the repository.


### 2. Model Weight
First you need to make sure that huggingface stays logged in.

```bash

huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir yourlocal_path

huggingface-cli download Divs1159/stingbee-7b --local-dir yourlocal_path

```
After downloading, you need to change the `"mm_vision_tower"` field in the `config.json` file under `stingbee-7b` to the local path of `openai/clip-vit-large-patch14-336`.
```bash
"mm_vision_tower":"yourlocal_path"
```
### 3. Eval
Benchmark questions: `stingbee/othermodel_eval/classfication_benchmark` <br>
You can use the following command to start the verification program directly:
```bash
bash stingbee/othermodel_eval/stingbee_multigpu_eval.sh
```
Parameters need to be set by yourself, different datasets save different jsonl files, the code to calculate F1 and mAP is as follows:
```bash
python stingbee/othermodel_eval/multidetect_cls_metric_cal.py --predictions xxx.jsonl --output xxx.txt #for multiclassification

python stingbee/othermodel_eval/singledetect_cls_metric_cal.py --predictions xxx.jsonl --output xxx.txt #for singleclassfication
```


