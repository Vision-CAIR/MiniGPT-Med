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
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 0
```

## Dataset
| Dataset | Images  | json file| 
|---------|---------|----------|
| MIMIC   |[Download](https://physionet.org/content/mimiciii/1.4/) | [Download](https://drive.google.com/drive/folders/1nZhdfNoh7fkx7CWvf0_47_OLv3tA2m3o?usp=sharing) |
| NLST    |[Download](https://wiki.cancerimagingarchive.net/display/NLST)| [Downlaod](https://drive.google.com/drive/folders/1OKgMTaGLu_dWRuco6JipYzezw3oNwgaz?usp=sharing) |
|SLAKE    |[Downlaod](https://www.med-vqa.com/slake/) |[Download](https://drive.google.com/drive/folders/1vstjmfRbKahSAsi_b6FmTQiuolvgO8oC?usp=sharing)|
|RSNA     |[Downlaod](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) | [Download](https://drive.google.com/drive/folders/1wkXPvUNqda6jWAIduyiVJkS3Tx7P7td8?usp=sharing) |
|Rad-VQA  |[Downalod](https://osf.io/89kps/) |[Download](https://drive.google.com/drive/folders/1ING6Dodwk2DU_t4GHQYudNFMMg9OMfBQ?usp=sharing) |

## Acknowledgement

- MiniGPT-4
- Lavis
- Vicuna
- Falcon
- Llama 2
