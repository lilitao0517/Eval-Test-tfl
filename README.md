# <img src="./images/logo.jpeg" height="150"> STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection [ CVPR-2025 ]

<p align="center">
  <strong>
    <a href="https://www.linkedin.com/in/divya-velayudhan-958052175">Divya Velayudhan</a>¹,  
    <a href="https://scholar.google.com/citations?user=2tHwtZwAAAAJ&hl=en">Abdelfatah Ahmed</a>¹, 
    <a href="https://www.linkedin.com/in/mohamad-alansari/">Mohamad Alansari</a>¹, 
    <a href="https://www.linkedin.com/in/neha-gour-3b501055/">Neha Gour</a>¹, 
    <a href="https://www.linkedin.com/in/abderaouf-behouch-2a1207102/">Abderaouf Behouch</a>¹,  
    <a href="https://www.linkedin.com/in/taimur-hassan-46a4a950/">Taimur Hassan</a>², 
    <a href="https://www.linkedin.com/in/wasimsyedtalal/">Syed Talal Wasim</a>³,⁴, 
    <a href="https://scholar.google.com/citations?user=Y0KW_J4AAAAJ&hl=en">Nabil Maalej</a>¹,  
    <a href="https://muzammal-naseer.com/">Muzammal Naseer</a>¹, 
    <a href="https://www.linkedin.com/in/juergen-gall-a78103204/">Juergen Gall</a>³,⁴,  
    <a href="https://www.linkedin.com/in/mohammed-bennamoun-b3147174/">Mohammed Bennamoun</a>⁵, 
    <a href="https://www.linkedin.com/in/ernestodamiani/">Ernesto Damiani</a>¹, 
    <a href="https://www.linkedin.com/in/naoufel-werghi-80846338/">Naoufel Werghi</a>¹  
  </strong>
</p>
  

<p align="center">
  ¹ Khalifa University of Science and Technology &emsp;&emsp;&emsp;&emsp;
  ² Abu Dhabi University &emsp;&emsp;&emsp;&emsp;
  <br>
  ³ University of Bonn &emsp;&emsp;&emsp;&emsp;
  ⁴ Lamarr Institute for ML and AI &emsp;&emsp;&emsp;&emsp;
  ⁵ The University of Western Australia
</p>

[![Website](https://img.shields.io/badge/STING--BEE-Website-87CEEB)](https://divs1159.github.io/STING-BEE/) [![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B)](https://arxiv.org/)  [![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/Divs1159/STING-BEE) [![Dataset](https://img.shields.io/badge/STCray-Dataset-228B22)](https://huggingface.co/datasets/Naoufel555/STCray-Dataset)

---

## 📢 Latest Updates
- **Apr-02-25**: STING-BEE paper is released [arxiv link]. 
- **Mar-25-25**: We open-source the code, model, dataset, and evaluation scripts. 
- **Feb-27-25**: STING-BEE has been accepted to **CVPR-25** 🎉.
  
---

## Contents
- [**Overview**](#overview)
- [**Highlights**](#highlights)
- [**Install**](#install)
- [**Model Weights**](#model-weights)

---  

## <img src="images/logo.jpeg" height="40">**Overview**  

Advancements in Computer-Aided Screening (CAS) systems are crucial for enhancing the detection of security threats in X-ray baggage scans. However, existing datasets fail to capture real-world, sophisticated threats and concealment tactics, while current models operate within a closed-set paradigm with predefined labels. To address these limitations, we introduce **STCray**, the **first multimodal X-ray baggage security dataset**, comprising **46,642 image-caption paired scans** across **21 threat categories**. Developed with a **specialized STING protocol**, STCray ensures **domain-aware, coherent captions**, enabling the creation of **multi-modal instruction-following data** for security screening applications.  

Leveraging **STCray**, we propose **STING-BEE**, the **first domain-aware visual AI assistant** for X-ray baggage security. **STING-BEE** unifies **scene comprehension, referring threat localization, visual grounding, and visual question answering (VQA)**, establishing **new benchmarks** for **multi-modal learning** in X-ray security research. Furthermore, it demonstrates **state-of-the-art generalization** across **cross-domain settings**, outperforming existing models in handling **real-world threat detection scenarios**.

---

## ✨**Highlights**

- [**STCray**](#stcray)
- [**STING-BEE**](#sting-bee)  

It establishes **new baselines** for **multimodal learning in X-ray baggage security**.  

---

## **STCray**  

We introduce STCray, the first X-ray baggage security dataset with 46,642 image-caption paired scans spanning 21 categories, including Improvised Explosive Devices (IEDs) and 3D-printed firearms. We meticulously develop STCray by carefully preparing and scanning baggage containing the threat and non-threat items to simulate a realistic environment, following our proposed STING protocol.

### Overview of the STCray dataset with real-world threats and image-text paired data

<div align="center">
  <img src="images/TopFig1.png" alt="STCray Dataset Overview" width="75%">
</div>

### **Comparison with Other X-ray Datasets**

| Dataset  | #Classes | Multimodal | Strategic Concealment | Emerging Novel Threats | Zero-shot Task |
|---------------|---------|------------|------------|------------|------------|
| GDXray (JNDE'15) | 3  | ❌ | ❌ | ❌ | ❌ |
| SIXray (CVPR'19) | 6  | ❌ | ❌ | ❌ | ❌ |
| OPIXray (ACMMM'20) | 5  | ❌ | ❌ | ❌ | ❌ |
| HiXray (ICCV'21) | 8  | ❌ | ❌ | ❌ | ❌ |
| DvXray (IEEE IFS'22) | 15 | ❌ | ❌ | ❌ | ❌ |
| CLCXray (IEEE IFS'22) | 12 | ❌ | ❌ | ❌ | ❌ |
| PIDRay (IJCV'23) | 15 | ❌ | ❌ | ❌ | ❌ |
| **STCray (Ours)** | **21** | ✅ | ✅ | ✅ | ✅ |

* Comparison based on multimodality, strategic concealment, novel threats, and zero-shot task capabilities.*

---
## **STING-BEE**

We introduce STING-BEE, the domain-aware visual AI assistant for X-ray baggage screening, trained on the instruction following dataset derived from the image-caption pairs of our proposed STCray dataset. STING-BEE provides a unified platform for scene comprehension, referring threat localization, visual grounding, and VQA, establishing new baselines for X-ray baggage security research.

<div align="center">
  <img src="images/STCray_Proposed_CVPR_V4.png" alt="STING-BEE Training and Evaluation Pipeline" width="100%">
  </div>
  <p class="absfont text-justify">(Left) STCray Dataset Collection, capturing X-ray images with systematic varia-
tions in threat type, location, and occlusion, along with detailed captions and bounding box annotations; (Center) Multi-modal Instruction
Tuning, consisting of Multi-task Threat Instruction Tuning and Threat Visual-Grounded Instruction Tuning (Right) Examples of STING-
BEE evaluation tasks including Scene Comprehension, Referring Expression, Visual Grounding, and VQA.</p>

---

## **Install**

1. Clone this repository and navigate to STING-BEE folder
```bash
git clone https://github.com/Divs1159/STING-BEE.git
cd STING-BEE
```

2. Install Package
```Shell
conda create -n stingbee python=3.9 -y
conda activate stingbee
pip install --upgrade pip  
pip install -e .

```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip uninstall transformers
pip install -e .
```
---

## 🔗**Model Weights**

Please check out our [Model Zoo](https://github.com/Divs1159/STING-BEE/blob/main/docs/MODEL_ZOO.md) for detailed information on STING-BEE checkpoint weights.

Alternatively, you can directly download the **STING-BEE-7B** model weights from [🤗 Hugging Face](https://huggingface.co/Divs1159/stingbee-7b).

Check [LoRA.md](https://github.com/Divs1159/STING-BEE/blob/main/docs/LoRA.md) for instructions on how to run the demo.


---

## Train

STING-BEE training consists of visual instruction tuning using StingBee_XrayInstruct data: Multimodal instruction-following data generated using STCray, fine-tuned over the pre-trained weights of LlaVA-v1.5.

We train STING-BEE on 2 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps`. To keep the global batch size the same, use the formula: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We used the following hyperparameters in fine-tuning:

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| STING-BEE-7B | 96 | 2e-5 | 1 | 2048 | 0 |

### Pretrain (feature alignment)

We use the pretrained projector from LLaVAv1.5, which is trained on 558K subset of the LAION-CC-SBU dataset with BLIP captions.

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.



## **📄 Citation**  

If you use **STING-BEE** in your research, please cite our work:  
