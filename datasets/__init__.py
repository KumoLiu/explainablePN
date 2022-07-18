from strix.utilities.registry import DatasetRegistry

CLASSIFICATION_DATASETS = DatasetRegistry()
SEGMENTATION_DATASETS = DatasetRegistry()
from .jsph_mvi_datasets import *
from .jsph_mps_datasets import *
from .lidc_idri_datasets import *