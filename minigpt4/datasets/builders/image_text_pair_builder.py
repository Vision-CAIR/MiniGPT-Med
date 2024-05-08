import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.mimic_cxr_dataset import MimicCxrDataset
from minigpt4.datasets.datasets.radvqa_dataset import RadVQADataset
from minigpt4.datasets.datasets.rsna_dataset import RSNADataset,ReferRSNADataset,IdentifyRSNADataset
from minigpt4.datasets.datasets.nlst_dataset import NlstDataset,ReferNLSTDataset,IdentifyNLSTDataset
from minigpt4.datasets.datasets.SLAKE_dataset import GroundingSLAKEDatase

@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets
    
@registry.register_builder("mimic_cxr")
class MimicCxrBuilder(BaseDatasetBuilder):
    train_dataset_cls = MimicCxrDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mimic_cxr/mimic_cxr.yaml",
    }

    def build_datasets(self):
        logging.info("Building MIMIC dataset...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )


        return datasets
    
@registry.register_builder("radvqa")
class RadVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = RadVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/radvqa/radvqa.yaml",
    }
    def build_datasets(self):
        logging.info("Building RADVQA datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )


        return datasets
    
@registry.register_builder("rsna")
class RSNABuilder(BaseDatasetBuilder):
    train_dataset_cls = RSNADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsna/rsna.yaml",
    }
    def build_datasets(self):
        logging.info("Building RSNA dataset...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )
        return datasets

@registry.register_builder("refer_rsna")
class ReferRSNABuilder(BaseDatasetBuilder):
    train_dataset_cls = ReferRSNADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refer_rsna/refer_rsna.yaml",
    }

    def build_datasets(self):
        logging.info("Building [refer] RSNA datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )
        return datasets
    
@registry.register_builder("identify_rsna")
class IdentifyRSNABuilder(BaseDatasetBuilder):
    train_dataset_cls = IdentifyRSNADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/identify_rsna/identify_rsna.yaml",
    }
    def build_datasets(self):
        logging.info("Building [identify] RSNA dataset...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )
        return datasets
    
    
@registry.register_builder("nlst")
class NlstBuilder(BaseDatasetBuilder):
    train_dataset_cls = NlstDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nlst/nlst.yaml",
    }
    def build_datasets(self):
        logging.info("Building NLST dataset...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets
    
@registry.register_builder("refer_nlst")
class ReferNLSTBuilder(BaseDatasetBuilder):
    train_dataset_cls = NlstDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/refer_nlst/refer_nlst.yaml",
    }
    def build_datasets(self):
        logging.info("Building [refer] NLST dataset...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets

@registry.register_builder("identify_nlst")
class IdentifyNLSTBuilder(BaseDatasetBuilder):
    train_dataset_cls = NlstDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/identify_nlst/identify_nlst.yaml",
    }
    def build_datasets(self):
        logging.info("Building [identify] NLST dataset...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets
    
@registry.register_builder("grounding_SLAKE")
class GroundingSLAKEBuilder(BaseDatasetBuilder):
    train_dataset_cls = GroundingSLAKEDatase
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/grounding_SLAKE/grounding_SLAKE.yaml",
    }

    def build_datasets(self):
        logging.info("Building [grounding] NLST dataset...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        dataset_cls = self.train_dataset_cls

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets



# @registry.register_builder("detect_mimic")
# class DetectMIMICBuilder(BaseDatasetBuilder):
#     train_dataset_cls = Detect_MIMIC
#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/detect_mimic/detect_mimic.yaml",
#     }
#     def build_datasets(self):
#         logging.info("Building NLST dataset...")
#         self.build_processors()
#         build_info = self.config.build_info
#         datasets = dict()

#         dataset_cls = self.train_dataset_cls

#         datasets['train'] = dataset_cls(
#             vis_processor=self.vis_processors['train'],
#             text_processor=self.text_processors['train'],
#             ann_path=build_info.ann_path,
#             vis_root=build_info.image_path,
#         )

#         return datasets
    

