import os
import json
from PIL import Image
from torch.utils.data import Dataset

class RadVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.instruction_pool = ["[vqa] {}"]
        
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
            
    def process_image(self, image_name):
        image_path = os.path.join(self.vis_root, image_name)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        return self.vis_processor(image)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        image = self.process_image(info['image_name'])
        instruction = self.text_processor(self.instruction_pool[0].format(info['question']))
        instruction = '[INST] <Img><ImageHere></Img> {} [/INST]'.format(instruction)

        answer = str(info['answer'])

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": info['image_name'],
        }
    
class evalRadVQADataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        info = self.loaded_data[idx]
        image_file = '{}'.format(info['image_name'])
        image_path = os.path.join(self.root_path, image_file)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        question = "[vqa] {}".format(info['question'])
        return image, question, image_file
