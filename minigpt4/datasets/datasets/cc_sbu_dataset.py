import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset
import json
import random
from webdataset import select


def process_bbox(phrases, boxes):
    new_boxes = []
    for box in boxes:
        small_box = []
        for ele in box:
            small_box.append(int(round(ele,2)*224))
        new_boxes.append(small_box)

    output = dict()

    for index,phrase in enumerate(phrases):
        box = new_boxes[index]
        if phrase not in output.keys():
            output[phrase]=[str(box)]
        else:
            output[phrase].append(str(box))

    full_sentence = ""
    for phrase in output.keys():
        if len(output[phrase])==1:
            bboxs = output[phrase][0]
            sentence = "{}: {} ".format(phrase,bboxs)
        else:
            if len(output[phrase]) >2:
                output[phrase] = random.sample(output[phrase],1)
            bboxs = ",".join(output[phrase])
            sentence = "{}: {} ".format(phrase,bboxs)
        full_sentence += sentence

    return full_sentence


def sample_phrase_box(phrases, boxes):
    new_boxes = []
    for box in boxes:
        small_box = []
        for ele in box:
            small_box.append(int(round(ele,2)*224))
        new_boxes.append(small_box)

    index = random.sample(range(0,len(phrases)),1)[0]
    return phrases[index], str(new_boxes[index])

def sample_phrase(phrases, region):
    # new_boxes = []
    # for box in boxes:
    #     small_box = []
    #     for ele in box:
    #         small_box.append(int(round(ele,2)*224))
    #     new_boxes.append(small_box)

    index = random.sample(range(0,len(phrases)),1)[0]

    return phrases[index], region[index]




class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.instruction_pool = [
            'Briefly describe this image.',
            'Provide a concise depiction of this image.',
            'Present a short description of this image.',
            'Summarize this image in a few words.',
            'A short image caption:',
            'A short image description:',
            'A photo of ',
            'An image that shows ',
            'Write a short description for the image. ',
            'Write a description for the photo.',
            'Provide a description of what is presented in the photo.',
            'Briefly describe the content of the image.',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.',
        ]

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        instruction = random.choice(self.instruction_pool)

        # instruction = "###Human: <Img><ImageHere></Img> {}###Assistant: ".format(instruction)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)

        return {
                "image": sample[0],
                "instruction_input": instruction,
                "answer": self.text_processor(sample[1]["caption"]),
            }


class CCSBUBBOXDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.bbox_json = json.load(open("/ibex/project/c2133/aa_shenx/GroundingDINO/cc_box_filter_new.json"))

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.select(self.filter_sample),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def filter_sample(self,sample):
        # print(sample[1]["key"] in self.bbox_json)
        return sample[1]["key"] in self.bbox_json

    def to_dict(self, sample):
            
        image_key = sample[1]["key"]

        phrases =  self.bbox_json[image_key]["phrases"]
        boxes = self.bbox_json[image_key]["boxes"]
        phrase_region = self.bbox_json[image_key]["box_regions"]

        phrase, region = sample_phrase(phrases,phrase_region)

        # phrase = " the bounding box of "+phrase+" is "
        # box = phrase+box

        phrase_input  = "Given an image, identify the objects and their bounding boxes in the format of {object, x1,y1,x2,y2}. "
        box_input = phrase_input + region

        return {
            "image": sample[0],
            "answer": self.text_processor(sample[1]["caption"]),
            "phrase_input": self.text_processor(phrase_input),
            "box_input": self.text_processor(box_input),
            "data_type": "bbox",
            "question_split": True
        }





class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        # if ann["image_id"] in self.bbox_json:
        #     print(ann["image_id"])
        # else:
        #     print("false")
        # assert False

        image = self.vis_processor(image)
        caption = ann["caption"]

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "data_type": "caption",
            "question_split": True
        }