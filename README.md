# MiniGPT-Rad
## Introduction
This project builds upon the foundations of MiniGPT4-V2, a large language model designed for vision-language multi-task learning. We have fine-tuned this model specifically for the analysis and interpretation of medical radiology images, aiming to bring significant advancements in the medical sector.

## Installation

```
git clone https://github.com/SDAIA-KAUST-AI/RadGPT
cd RadGPT
conda env create -f environment.yml
conda activate RadGPT
```

## Download RadGPT trained weights

**Last trained weight with only public datasets and Llama2 could be downloaded at [radgpt4_llama2_7bchat_stage3.pth](https://drive.google.com/file/d/1wE-uIS2meGGt6z8LbqUVwHBPQsE4rd3B/view).

** Then modify line 8 at RadGPT/eval_configs/minigptv2_eval.yaml to be the path of RadGPT weight.

## Prepare weight for LLMs

### Llama2 Version

```shell
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```

Then modify line 14 at RadGPT/minigpt4/configs/models/minigpt_v2.yaml to be the path of Llama-2-13b-chat-hf.

## Launching Demo Locally

```
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0
```

## Acknowledgement

- [MiniGPT-4](https://minigpt-4.github.io/) This repo is developped on MiniGPT-4, an awesome repo for vision-language chatbot!
- Lavis
- Vicuna
- Falcon
- Llama 2
