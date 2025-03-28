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
- **Apr-02-25**: STING-BEE paper is released [arxiv link]. 🔥🔥
- **Mar-25-25**: We open-source the code, model, dataset, and evaluation scripts. 
- **Feb-27-25**: STING-BEE has been accepted to **CVPR-25** 🎉.
  
---

## <img src="images/logo.jpeg" height="40">Overview  

**STING-BEE** is the first **domain-aware visual AI assistant** for X-ray baggage security screening. It is trained on **STCray**, the first multimodal X-ray baggage security dataset, comprising **46,642 image-caption paired scans** spanning **21 threat categories**, including **novel threats** such as **Improvised Explosive Devices (IEDs)** and **3D-printed firearms**. STING-BEE serves as a unified platform for scene comprehension, referring threat localization, visual grounding, and visual question answering (VQA), establishing new benchmarks for X-ray baggage security research.  

---

## **✨ Highlights**

- [**STCray**](#stcray)
- [**STING-BEE**](#sting-bee)  

It establishes **new baselines** for **multimodal learning in X-ray baggage security**.  

---

## **STCray**  

### Overview of the STCray dataset with real-world threats and image-text paired data

<div align="center">
  <img src="images/TopFig1.png" alt="STCray Dataset Overview" width="75%">
</div>

### **Comparison with Other X-ray Datasets**

| Dataset  | #Classes | Multimodal | Strategic Concealment | Emerging Novel Threats | Zero-shot Task |
|----------|---------|------------|-----------------------|----------------------|----------------|
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



## **📄 Citation**  

If you use **STING-BEE** in your research, please cite our work:  
