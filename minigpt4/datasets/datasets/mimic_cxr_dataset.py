import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset

class MimicCxrDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
        
        self.instruction_pool = [
            'Describe this image in detail',
            'Take a look at this image and describe what you notice',
            'Please provide a detailed description of the picture',
            'Could you describe the contents of this image for me?'
        ]

    def load_image(self, image_id):
        image_file = f'{image_id}.jpg'
        image_path = os.path.join(self.vis_root, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        return image

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        image = self.load_image(info['image_id'])
        instruction = random.choice(self.instruction_pool)
        instruction = f'<Img><ImageHere></Img> {self.text_processor(instruction)}'

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": info['caption'],
            "image_id": info['image_id'],
        }


# class Detect_MIMIC(Dataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_path):
#         self.vis_root = vis_root
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor

#         with open(ann_path, 'r') as f:
#             self.ann = json.load(f)

#         self.original_size = 1024
#         self.image_size = 100

#         # Extract object names from the annotation file and create the instruction pool
#         self.instruction_pool = []
#         for annotation in self.ann:
#             if 'objects' in annotation and annotation['objects']:
#                 self.instruction_pool.extend(['[detection] ' + obj for obj in annotation['objects']])

#     def __len__(self):
#         return len(self.ann)
    
#     def __getitem__(self, index):
#         return self.bbox_phrase_preprocess(index)
    
#     def prepare_image_and_annotations(self, info):
#         image = self.process_image(info["key"])
#         bboxs, ref_phrases = self.generate_bboxs_and_phrases(info)
#         return image, bboxs, ref_phrases

#     def process_image(self, image_file):
#         image_file = '{}.png'.format(image_file)
#         image_path = os.path.join(self.vis_root, image_file)
#         grayscale_image = Image.open(image_path).convert("L")
#         image = Image.new("RGB", grayscale_image.size)
#         image.paste(grayscale_image)
#         return self.vis_processor(image)
    
#     def generate_bboxs_and_phrases(self, info):
#         bboxs, ref_phrases = [], []
#         for bbox in info["bbox"]:
#             scaled_bbox = self.scale_bbox(*bbox)
#             self.assert_bbox_in_range(*scaled_bbox)
#             ref_phrases.append("tumor")
#             bboxs.append(f"{{<{scaled_bbox[0]}><{scaled_bbox[1]}><{scaled_bbox[2]}><{scaled_bbox[3]}>}}")
#         return bboxs, ref_phrases
    
#     def scale_bbox(self, x1, y1, x2, y2):
#         scale = lambda x: int((x / self.original_size) * self.image_size)
#         return scale(x1), scale(y1), scale(x2), scale(y2)

#     def assert_bbox_in_range(self, x1, y1, x2, y2):
#         for coord in [x1, y1, x2, y2]:
#             assert 0 <= coord <= self.image_size, f"{coord} out of range"
            
#     def generate_caption(self, phrases, bounding_boxes):
#         phrase_bbox={}
#         for phrase, bbox in zip(phrases, bounding_boxes):
#             if phrase not in phrase_bbox.keys():
#                 generated_phrase = "<p>{}</p> ".format(phrase)
#                 generated_phrase_bbox = generated_phrase+str(bbox)
#             else:
#                 generated_phrase = phrase_bbox[phrase]
#                 generated_phrase_bbox = generated_phrase+"<delim>"+str(bbox)
#             phrase_bbox[phrase] = generated_phrase_bbox
#         grounded_caption= ' '.join(phrase_bbox.values())
#         return grounded_caption

#     def bbox_phrase_preprocess(self, index):
#         info = self.ann[index]
#         image, bboxs, ref_phrases = self.prepare_image_and_annotations(info)

#         generated_caption = self.generate_caption(ref_phrases, bboxs)
#         instruction = f'[INST] <Img><ImageHere></Img> {self.instruction_pool[0]} [/INST]'

#         return {
#             "image": image,
#             "instruction_input": instruction,
#             "answer": generated_caption,
#             "image_id": info['key'],
#         }


#####Eval Classes#####

class evalMIMICDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

        self.instruction_pool = [
            'Describe this image in detail',
            'Take a look at this image and describe what you notice',
            'Please provide a detailed description of the picture',
            'Could you describe the contents of this image for me?'
        ]

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        info = self.loaded_data[idx]
        img_id = '{}.jpg'.format(info['image_id'])
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)

        answer = info['caption']
        question = random.choice(self.instruction_pool)

        return image, question, img_id
    
    
class evalDetectMimicDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['key']
        sent = data['objects']
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        question = f"[detection] {sent}"

        return image, question, img_id