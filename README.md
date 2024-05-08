# MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis
Asma Alkhaldi, Raneem Alnajim, Layan Alabdullatef, Rawan Alyahya, Jun Chen, Deyao Zhu, Ahmed Alsinan, Mohamed Elhoseiny
Saudi Data and Artificial Intelligence Authority (SDAIA)
King Abdullah University of Science and Technology (KAUST)

<a href='MiniGPT-Med.pdf'><img src='paper Link></a>

## Installation
```
conda env create -f environment.yml
conda activate miniGPT-Med
```

## Download miniGPT-Med trained weights

**Last trained weight with only public datasets and Llama2 could be downloaded at [miniGPT-Med.pth](https://drive.google.com/file/d/18C5KkAkdsW04IMnKX8s_HaL__f8Zlf7B/view?usp=sharing).

** Then modify line 8 at miniGPT-Med/eval_configs/minigptv2_eval.yaml to be the path of miniGPT-Med weight.

## Prepare weight for LLMs

### Llama2 Version

```shell
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```

Then modify line 14 at miniGPT-Med/minigpt4/configs/models/minigpt_v2.yaml to be the path of Llama-2-13b-chat-hf.

## Launching Demo Locally

```
python demo.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 0
```

## Acknowledgement

- [MiniGPT-4](https://minigpt-4.github.io/) This repo is developped on MiniGPT-4, an awesome repo for vision-language chatbot!
- Lavis
- Vicuna
- Falcon
- Llama 2
