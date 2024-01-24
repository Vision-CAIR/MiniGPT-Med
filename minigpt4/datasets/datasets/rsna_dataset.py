import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset

class RSNADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        self.original_size = 1024
        self.image_size = 100
        self.instruction_pool = ['[detection] pneumonia']

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        return self.bbox_phrase_preprocess(index)

    def prepare_image_and_annotations(self, info):
        image = self.process_image(info["key"])
        bboxs, ref_phrases = self.generate_bboxs_and_phrases(info)
        return image, bboxs, ref_phrases

    def process_image(self, image_file):
        image_path = os.path.join(self.vis_root, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        return self.vis_processor(image)

    def generate_bboxs_and_phrases(self, info):
        bboxs, ref_phrases = [], []
        for bbox in info["bbox"]:
            scaled_bbox = self.scale_bbox(*bbox)
            self.assert_bbox_in_range(*scaled_bbox)
            ref_phrases.append("pneumonia")
            bboxs.append(f"{{<{scaled_bbox[0]}><{scaled_bbox[1]}><{scaled_bbox[2]}><{scaled_bbox[3]}>}}")
        return bboxs, ref_phrases

    def scale_bbox(self, x1, y1, x2, y2):
        scale = lambda x: int((x / self.original_size) * self.image_size)
        return scale(x1), scale(y1), scale(x2), scale(y2)

    def assert_bbox_in_range(self, x1, y1, x2, y2):
        for coord in [x1, y1, x2, y2]:
            assert 0 <= coord <= self.image_size, f"{coord} out of range"

    def generate_caption(self, phrases, bounding_boxes):
        phrase_bbox={}
        for phrase, bbox in zip(phrases, bounding_boxes):
            if phrase not in phrase_bbox.keys():
                generated_phrase = "<p>{}</p> ".format(phrase)
                generated_phrase_bbox = generated_phrase+str(bbox)
            else:
                generated_phrase = phrase_bbox[phrase]
                generated_phrase_bbox = generated_phrase+"<delim>"+str(bbox)
            phrase_bbox[phrase] = generated_phrase_bbox
        generated_caption= ' '.join(phrase_bbox.values())
        return generated_caption
    
    def bbox_phrase_preprocess(self, index):
        info = self.ann[index]
        image, bboxs, ref_phrases = self.prepare_image_and_annotations(info)

        generated_caption = self.generate_caption(ref_phrases, bboxs)
        instruction = f'[INST] <Img><ImageHere></Img> {self.instruction_pool[0]} [/INST]'

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": generated_caption,
            "image_id": info['key'],
        }

class ReferRSNADataset(RSNADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, dataset='refcoco', splitBy='unc'):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.instruction_pool = [
            "[refer] pneumonia"
            "[refer] give me the location of pneumonia",
            "[refer] where is pneumonia ?",
            "[refer] from this image, tell me the location of pneumonia",
            "[refer] the location of pneumonia is ",
            "[refer] could you tell me the location for pneumonia ?",
            "[refer] where can I locate the pneumonia ?",
        ]

    def bbox_phrase_preprocess(self, index):
        info = self.ann[index]
        image, bboxs, ref_phrases = self.prepare_image_and_annotations(info)

        generated_caption = self.generate_caption(ref_phrases, bboxs)
        instruction = '[INST] <Img><ImageHere></Img> {} [/INST]'.format(random.choice(self.instruction_pool))

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": generated_caption,  
            "image_id": info['key'],
        }

class IdentifyRSNADataset(RSNADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.instruction_pool = [
            "[identify] {}",
            "[identify] what object is in this location {}",
            "[identify] identify the object present at this location {}",
            "[identify] what is it in {}",
            "[identify] describe this object in {}",
            "[identify] this {} is",
            "[identify] the object in {} is",
        ]
    
    def generate_boxes(self, phrases, bounding_boxes):
        phrase_bbox = {}
        for phrase, bbox in zip(phrases, bounding_boxes):
            if phrase not in phrase_bbox:
                grounded_bbox = str(bbox)
            else:
                grounded_bbox = phrase_bbox[phrase] + "<delim>" + str(bbox)
            phrase_bbox[phrase] = grounded_bbox
        
        ground_boxes = ' '.join(phrase_bbox.values())
        return ground_boxes

    def bbox_phrase_preprocess(self, index):
        info = self.ann[index]
        image = self.process_image(info['key'])
        ref_exps = info["bbox"]
        caption = info["rephrased_caption"]
        bboxs, ref_phrases = self.generate_bboxs_and_phrases(info)
        identify_boxes = self.generate_boxes(ref_phrases,bboxs)
        instruction = random.choice(self.instruction_pool).format(identify_boxes)
        instruction = f'[INST] <Img><ImageHere></Img> {instruction} [/INST]'

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": caption,
            "image_id": info['key'],
        }


class evalRSNADataset(Dataset):
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
        question = "[detection] pneumonia"

        return image, question, img_id