# Classification Evaluation and Captioner on Pidray and OPIXray

The code is built on the official STING-BEE repository, due to the lack of official requirements, please be sure to follow the steps below to configure the environment. In addition, we have built `Onepipe` semi-automated caption fetch pipeline in this repository as well.

---
## 📅 Important News
We are about to present a unified X-ray visual language model that achieves excellent performance on major benchmarks. We will provide a Comprehensive multimodal Benchmark for X-ray contraband. The paper, dataset, code, and weights will be made public after the paper is accepted.

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
conda create -n msswift python=3.9
conda activate msswift
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.31.0 #if you want to
```

For the rest of the configuration we recommend that you follow the configuration requirements of ms-swift exactly, the mandatory packages are as follows:
|              | Range        | Recommended         | Notes                                     |
|--------------|--------------|---------------------|-------------------------------------------|
| python       | >=3.9        | 3.10/3.11                |                                           |
| cuda         |              | cuda12              | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        | 2.7.1               |                                           |
| transformers | >=4.33       | 4.56.2              |                                           |
| modelscope   | >=1.23       |                     |                                           |
| peft         | >=0.11,<0.18 |                     |                                           |
| flash_attn   |              | 2.8.1/3.0.0b1 |                                           |
| trl          | >=0.15,<0.24 | 0.20.0              | RLHF                                      |
| deepspeed    | >=0.14       | 0.17.5              | Training                                  |
| vllm         | >=0.5.1      | 0.10.1.1                | Inference/Deployment                      |
| sglang       | >=0.4.6      | 0.4.10.post2         | Inference/Deployment                      |
| lmdeploy     | >=0.5   | 0.9.2.post1                 | Inference/Deployment                      |
| evalscope    | >=1.0       |                     | Evaluation                                |
| gradio       |              | 5.32.1              | Web-UI/App                                |



### 2. Model Weight
First you need to make sure that huggingface stays logged in.<br>
If you want to use STING-BEE, you should:
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


### 4. Getting the Caption
We have obtained rough captions for opixray in `dataset/opixray/trainset_caption_coordinate_alignment.jsonl`<br>

We use `Qwen2.5-VL-7B` to get the caption, but of course, this model can be seamlessly integrated into other models. Simply modify the `local_model_path`.<br>

In order to obtain a better representation of the title and already better command compliance, we recommend the use of the `Qwen2.5-VL-32B-Instruct`.

To get Model:
```bash
modelscope download --model Qwen/Qwen2.5-VL-32B-Instruct --local_dir yourlocal_path
```

For Opixray:
```bash
torchrun --nproc_per_node=NUM_GPUS dataset_construction/opixray/text_caption.py
```


For PIDray：
```bash
torchrun --nproc_per_node=NUM_GPUS dataset_construction/pidray/text_caption.py
```


