'''
use this command in terminal to run the evaluation script
torchrun --master-port 8888 --nproc_per_node 1 eval_scripts/model_evaluation.py  --cfg-path eval_configs/minigptv2_benchmark_evaluation.yaml --dataset 


'''

import sys
sys.path.append('.')
import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

from minigpt4.datasets.datasets.mimic_cxr_dataset import evalMIMICDataset, evalDetectMimicDataset
from minigpt4.datasets.datasets.radvqa_dataset import evalRadVQADataset
from minigpt4.datasets.datasets.nlst_dataset import eval_NLST_Dataset
from minigpt4.datasets.datasets.rsna_dataset import evalRSNADataset
from minigpt4.datasets.datasets.SLAKE_dataset import evalSLAKEDataset
#import cleaning classes
from eval_scripts.clean_json import clean_mimic_json, clean_vqa_json, clean_detection_json
from eval_scripts.metrics import MIMIC_BERT_Sim, VQA_BERT_Sim, average_iou

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, help="dataset to evaluate")

args = parser.parse_args()

cfg = Config(args)


model, vis_processor = init_model(args)
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path

def process_mimic_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open((eval_file_path), 'r') as f:
        mimic = json.load(f)
            
    data = evalMIMICDataset(mimic, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            minigpt4_predict[img_id].append(answer)
    
    file_save_path = os.path.join(save_path,"MIMIC_inference_results_stage3.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    clean_mimic_json(file_save_path, file_save_path)

    # csv file path to save the BERT results per each case
    output_csv_path = '/miniGPT-Med/metric_results/bert_similarity_scores.csv'

    # in MIMIC_BERT_Sim add the path of the ground_truth then the path of the inference result
    average_similarity = MIMIC_BERT_Sim(eval_file_path, file_save_path, output_csv_path)
    #print the average BERT_Sim
    print("Average BERT Similarity:", average_similarity)

def process_vqa_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open((eval_file_path), 'r') as f:
        radVQA = json.load(f)
            
    data = evalRadVQADataset(radVQA, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            minigpt4_predict[img_id].append({"key":img_ids,"question": question.replace("[vqa]", "").strip() , "answer": answer})
    
    file_save_path = os.path.join(save_path,"radVQA_inference_results.json")
    output_csv_path = '/miniGPT-Med/BERT_Sim_results/vqa_bert_similarity_scores.csv'

    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

    clean_vqa_json(file_save_path, file_save_path)
    VQA_BERT_Sim(eval_file_path, file_save_path, output_csv_path)

def process_nlst_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"] 
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open((eval_file_path), 'r') as f:
        nlst = json.load(f)

    data = eval_NLST_Dataset(nlst, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []

    for images, questions, img_ids in tqdm(eval_dataloader):

        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, img_id, question in zip(answers, img_ids, questions):

            # answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,2}><\d{1,2}><\d{1,2}><\d{1,2}>\}'
            minigpt4_predict[img_id].append(answer)

    file_save_path = os.path.join(save_path,"NLST_inference_result.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    
    csv_pth = os.path.join(save_path,"NLST_IoU_results.csv")
    clean_detection_json(file_save_path,file_save_path)
    average_iou(eval_file_path, file_save_path, 512, 100, "NLST", csv_pth)



def process_rsna_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    print(eval_file_path)
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"] 
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]
    print("----config----")
    with open((eval_file_path), 'r') as f:
        nlst = json.load(f)

    data = evalRSNADataset(nlst, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, img_id, question in zip(answers, img_ids, questions):

            # answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,2}><\d{1,2}><\d{1,2}><\d{1,2}>\}'
            minigpt4_predict[img_id].append(answer)
            print(img_id)
            print(answer)

    file_save_path = os.path.join(save_path,"RSNA_inference_result.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    
    csv_pth = os.path.join(save_path,"RSNA_IoU_results.csv")
    clean_detection_json(file_save_path,file_save_path)
    average_iou(eval_file_path, file_save_path, 1024, 100, "rsna", csv_pth)


def process_detect_mimic():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"] 
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open((eval_file_path), 'r') as f:
        nlst = json.load(f)

    data = evalDetectMimicDataset(nlst, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []

    for images, questions, img_ids in tqdm(eval_dataloader):

        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, img_id, question in zip(answers, img_ids, questions):
            pattern = r'\{<\d{1,2}><\d{1,2}><\d{1,2}><\d{1,2}>\}'
            minigpt4_predict[img_id].append(answer)

    file_save_path = os.path.join(save_path,"Detect_MIMIC_inference_result.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    

    csv_pth = os.path.join(save_path,"MIMIC_IoU_results.csv")
    clean_detection_json(file_save_path,file_save_path)
    average_iou(eval_file_path, file_save_path, "to be specified soon", 100, "MIMIC", csv_pth)



def process_SLAKE_dataset():
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"] 
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    with open((eval_file_path), 'r') as f:
        SLAKE = json.load(f)

    data = evalSLAKEDataset(SLAKE, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []

    for images, questions, img_ids in tqdm(eval_dataloader):

        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, img_id, question in zip(answers, img_ids, questions):

            # answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,2}><\d{1,2}><\d{1,2}><\d{1,2}>\}'
            minigpt4_predict[img_id].append(answer)

    file_save_path = os.path.join(save_path,"SLAKE_inference_result.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    
    csv_pth = os.path.join(save_path,"SLAKE_IoU_results.csv")
    clean_detection_json(file_save_path,file_save_path)
    average_iou(eval_file_path, file_save_path, 100, 100, "SLAKE", csv_pth)
    


############################################################################
for dataset in args.dataset:
    if dataset == 'mimic_cxr':
        process_mimic_dataset()

    elif dataset == 'radvqa':
        process_vqa_dataset()

    elif dataset == 'nlst':
        process_nlst_dataset()

    elif dataset == 'rsna':
        process_rsna_dataset()

    elif dataset == 'detect_mimic':
        process_detect_mimic()

    elif dataset == 'SLAKE':
        process_SLAKE_dataset()

    else:
        print(f"Dataset '{dataset}' is not supported.")