# MiniGPT-Med: Large Language Model as a General Interface for Radiology Diagnosis
Asma Alkhaldi, Raneem Alnajim, Layan Alabdullatef, Rawan Alyahya, Jun Chen, Deyao Zhu, Ahmed Alsinan, Mohamed Elhoseiny

*Saudi Data and Artificial Intelligence Authority (SDAIA) and King Abdullah University of Science and Technology (KAUST)*

## Installation
```
git clone https://github.com/Vision-CAIR/MiniGPT-Med
cd MiniGPT-Med
conda env create -f environment.yml
conda activate miniGPT-Med
```

## Download miniGPT-Med trained model weights

* miniGPT-Med's weights [miniGPT-Med Model](https://drive.google.com/file/d/18C5KkAkdsW04IMnKX8s_HaL__f8Zlf7B/view?usp=sharing)

* Then modify line 8 at miniGPT-Med/eval_configs/minigptv2_eval.yaml to be the path of miniGPT-Med weight.

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

## Dataset
| Dataset | Images  | json file| 
|---------|---------|----------|
| MIMIC   |[Download](https://physionet.org/content/mimiciii/1.4/) | [Download]() |
| NLST    |[Download](https://wiki.cancerimagingarchive.net/display/NLST)| [Downlaod]() |
|SLAKE    |[Downlaod](https://www.med-vqa.com/slake/) |[Download]()|
|RSNA     |[Downlaod](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) | [Download]() |
|Rad-VQA  |[Downalod](https://osf.io/89kps/) |[Download]() |

## Acknowledgement

- MiniGPT-4
- Lavis
- Vicuna
- Falcon
- Llama 2
