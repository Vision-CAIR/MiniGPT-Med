import sys
sys.path.append('.')

import json
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer, util
from minigpt4.common.eval_utils import computeIoU

# Load pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# BERT similarity function will be utilized in the two following functions
def compute_bert_similarity(prediction_caption, ground_truth_caption):
    prediction_embedding = model.encode([prediction_caption])
    ground_truth_embedding = model.encode([ground_truth_caption])
    similarity = util.pytorch_cos_sim(prediction_embedding, ground_truth_embedding)[0][0].item()
    return similarity


def MIMIC_BERT_Sim(gt_pth, pred_pth, output_csv):
    # Read the ground truth and prediction JSON files
    with open(gt_pth, 'r') as f:
        ground_truth_data = json.load(f)
    
    with open(pred_pth, 'r') as f:
        prediction_data = json.load(f)
    
    # Create a list to store BERT similarity data
    bert_similarity_data = []
    
    # Initialize variables to calculate the average
    total_similarity = 0
    total_count = 0
    
    # Iterate over each item in the prediction_data list
    for item in prediction_data:
        # Extract the image_id and corresponding prediction caption
        image_id = item["image_id"]
        prediction_caption = item["caption"]
        
        # Search for the matching ground truth caption based on image_id
        ground_truth_caption = None
        for gt_item in ground_truth_data:
            if gt_item["image_id"] == image_id:
                ground_truth_caption = gt_item["caption"]
                break
        
        if ground_truth_caption is not None:
            bert_similarity = compute_bert_similarity(prediction_caption, ground_truth_caption)
            bert_similarity_data.append({"image_id": image_id, "BERT_score": bert_similarity})
            
            total_similarity += bert_similarity
            total_count += 1
    
    average_similarity = total_similarity / total_count if total_count > 0 else 0
    
    df = pd.DataFrame(bert_similarity_data)
    df_sorted = df.sort_values(by="BERT_score", ascending=True)
    df_sorted.to_csv(output_csv, index=False)
    
    return average_similarity

def VQA_BERT_Sim(gt_pth, pred_pth, output_csv):
    # Load ground truth JSON file
    with open(gt_pth, 'r') as file:
        gt_data = json.load(file)

    # Load prediction JSON file
    with open(pred_pth, 'r') as file:
        prediction_data = json.load(file)

    gt_qa_pairs = {(entry['image_name'], entry['question']): entry['answer'] for entry in gt_data}

    def convert_to_dict(data):
        qa_dict = {}
        for image_name, qa_list in data.items():
            for qa in qa_list:
                key = (image_name, qa['question'])
                qa_dict[key] = qa['answer']
        return qa_dict

    pred_qa_dict = convert_to_dict(prediction_data)

    # Compute BERT similarity and create a list of results
    results = []

    for key, gt_answer in gt_qa_pairs.items():
        if key in pred_qa_dict:
            pred_answer = pred_qa_dict[key]
            gt_answer = str(gt_answer)
            pred_answer = str(pred_answer)

            # Compute BERT similarity
            similarity_score = compute_bert_similarity(pred_answer, gt_answer)

            # Append the result to the list
            results.append({
                "img_name": key[0],
                "question": key[1],
                "answer": pred_answer,
                "BERT_score": similarity_score
            })

    average_similarity = sum(entry["BERT_score"] for entry in results) / len(results) if results else 0
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="BERT_score", ascending=True)
    df_sorted.to_csv(output_csv, index=False)
    print(f"Average BERT similarity score: {average_similarity}")


#################################
##############IoU################
#################################

def preprocess_bbox(bbox, original_size, image_size):
    x1 = int((bbox[0] / original_size) * image_size)
    y1 = int((bbox[1] / original_size) * image_size)
    x2 = int((bbox[2] / original_size) * image_size)
    y2 = int((bbox[3] / original_size) * image_size)
    return [x1, y1, x2, y2]

def average_iou(gt_pth, pred_pth, original_size, image_size, dataset_name, csv_filename):
    # Load ground truth
    with open(gt_pth, 'r') as file:
        ground_truth = json.load(file)

    # Load predictions
    with open(pred_pth, 'r') as file:
        predictions = json.load(file)

    iou_list = []

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'IoU']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for gt_item in ground_truth:
            gt_key = gt_item['key']
            gt_bboxes = gt_item['bbox']
            original_size = gt_item['height']
            gt_processed_bboxes = [preprocess_bbox(bbox, original_size, image_size) for bbox in gt_bboxes]

            for pred_item in predictions:
                pred_key = pred_item['key'].replace(".png", "")

                if gt_key == pred_key:
                    pred_bboxes = pred_item['bbox']
                    try:
                        for gt_bbox in gt_processed_bboxes:
                            for pred_bbox in pred_bboxes:
                                iou = computeIoU(gt_bbox, pred_bbox)
                                iou_list.append(iou)
                                writer.writerow({'image_name': gt_key, 'IoU': iou})
                                print(gt_key)
                                print(iou)
                    except Exception as e:
                        print("gt_bbox: ", gt_bbox)
                        print("gt_bbox: ", pred_bboxes)

    # average_iou = sum(iou_list) / len(iou_list)
    # print(f"Average IoU for dataset {dataset_name}: {average_iou:.4f}")