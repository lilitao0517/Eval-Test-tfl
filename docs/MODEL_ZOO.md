# Model Zoo

| Base LLM | Vision Encoder | Pretrain Data | Pretraining schedule | Finetuning Data |  Download |
|----------|----------------|---------------|----------------------|-----------------|-----------|
| Vicuna-13B-v1.3 | CLIP-L-336px| LCS-558K | 1e | [StingBee_XrayInstruct](https://huggingface.co/datasets/Divs1159/StingBee_XrayInstruct) | [LoRA-Merged](https://huggingface.co/Divs1159/stingbee-7b) |

## Projector weights
We use the projector from LlaVA-1.5 for initialization. [Link](https://huggingface.co/liuhaotian/llava-v1.5-7b-lora)

**NOTE**: When you use our pre-trained projector for visual instruction tuning, it is very important to use the same base LLM and trained vision encoder. Otherwise, the performance may be very bad.

When using these projector weights to instruction-tune your LLM, please make sure that these options are correctly set as follows,

```Shell
--mm_use_im_start_end False
--mm_use_im_patch_token False
```
