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