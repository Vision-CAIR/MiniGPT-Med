import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset

# class SLAKEDataset(Dataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_path):
#         self.vis_root = vis_root
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor

#         with open(ann_path, 'r') as f:
#             self.ann = json.load(f)

#         self.original_size = 512
#         self.image_size = 100
#         # self.instruction_pool = ['[detection] pneumonia']
#         self.instruction_pool = list(set(item["object"] for item in self.ann))

#     def __len__(self):
#         return len(self.ann)
    
#     def __getitem__(self, index):
#         return self.bbox_phrase_preprocess(index)
    
#     def prepare_image_and_annotations(self, info):
#         image = self.process_image(info["key"])
#         bboxs, ref_phrases = self.generate_bboxs_and_phrases(info)
#         return image, bboxs, ref_phrases

#     def process_image(self, image_file):
#         image_file = '{}'.format(image_file)
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
#             # ref_phrases.append("tumor")
#             ref_phrases.append(f"{self.instruction_pool}")
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

# class GroundingSLAKEDatase(SLAKEDataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_path):
#         super().__init__(vis_processor, text_processor, vis_root, ann_path)
        
#         self.instruction_pool = [
#             '[grounding] describe this image in detail',
#             '[grounding] take a look at this image and describe what you notice',
#             '[grounding] please provide a detailed description of the picture',
#             '[grounding] could you describe the contents of this image for me?'
#         ]

#     def generate_caption(self,image_caption, phrases, bounding_boxes):
#         phrase_bbox={}
#         for phrase, bbox in zip(phrases, bounding_boxes):
#             if phrase not in phrase_bbox.keys():
#                 generated_phrase = "<p>{}</p> ".format(phrase)
#                 generated_phrase_bbox = generated_phrase+str(bbox)
#             else:
#                 generated_phrase = phrase_bbox[phrase]
#                 generated_phrase_bbox = generated_phrase+"<delim>"+str(bbox)
#             phrase_bbox[phrase] = generated_phrase_bbox
#         image_caption = image_caption.replace(phrase, generated_phrase_bbox)
#         return image_caption

#     def bbox_phrase_preprocess(self, index):
#         info = self.ann[index]
#         image = self.process_image(info['key'])
#         caption = info["caption"]
#         ref_exps = info["bbox"]

#         bboxs, ref_phrases = [], []
#         for item in ref_exps:
#             scaled_bbox = self.scale_bbox(*item)
#             self.assert_bbox_in_range(*scaled_bbox)
#             ref_phrases.append(f"{self.instruction_pool}")
#             bboxs.append(f"{{<{scaled_bbox[0]}><{scaled_bbox[1]}><{scaled_bbox[2]}><{scaled_bbox[3]}>}}")

#         generated_caption = self.generate_caption(caption, ref_phrases, bboxs)
#         instruction = random.choice(self.instruction_pool)
#         instruction = f'<Img><ImageHere></Img> {self.text_processor(instruction)}'

#         return {
#             "image": image,
#             "instruction_input": instruction,
#             "answer": generated_caption,
#             "image_id": info['key'],
#         }
    

class GroundingSLAKEDatase(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            '[grounding] please describe this image in details',
            '[grounding] describe this image as detailed as possible',
            '[grounding] summarize this image in details',
            '[grounding] give a thorough description of what you see in this image',
        ]

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = info['folder_name']
        image_path = os.path.join(self.vis_root, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
    
        answer = info['grounded_caption']

        instruction = random.choice(self.instruction_pool)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['folder_name'],
        }