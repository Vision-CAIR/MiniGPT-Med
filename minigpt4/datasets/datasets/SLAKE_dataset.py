import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset

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


class evalSLAKEDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['folder_name']
        # sent = data['objects']
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        question = "[grounding] please describe this image in details"

        return image, question, img_id