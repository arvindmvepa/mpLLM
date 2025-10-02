from abc import abstractmethod, ABC
from data.textbook_dataset import TextbookDataset
from data.vqa_dataset import NephVQADataset, NephVQAInstructFinetuneDataset, SlakeVQADataset, SlakeLlavaVQADataset, \
    Brats3DVQADataset, MimicVQADataset, ROCCOv2VQADataset, Brats3D2DVQADataset
from data.finetune_dataset import MimicFinetuneDataset, KGFinetuneDataset
from data.kg_dataset import SlakeKG


class DatasetFactory(ABC):
    """Abstract class for ModelFactory"""

    @abstractmethod
    def create_dataset(self, **dataset_params):
        raise NotImplementedError()


class TextbookDatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return TextbookDataset(**dataset_params)


class NephVQADatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return NephVQADataset(**dataset_params)


class NephVQAInstructFinetuneDatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return NephVQAInstructFinetuneDataset(**dataset_params)


class SlakeVQADatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return SlakeVQADataset(**dataset_params)


class SlakeLlavaVQADatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return SlakeLlavaVQADataset(**dataset_params)


class Brats3DVQADatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return Brats3DVQADataset(**dataset_params)

class Brats3D2DVQADatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return Brats3D2DVQADataset(**dataset_params)


class MimicVQADatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return MimicVQADataset(**dataset_params)


class MimicFinetuneDatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return MimicFinetuneDataset(**dataset_params)


class ROCCOv2VQADatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return ROCCOv2VQADataset(**dataset_params)


class KGFinetuneDatasetFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return KGFinetuneDataset(**dataset_params)


class SlakeKGFactory(DatasetFactory):
    def create_dataset(self, **dataset_params):
        return SlakeKG(**dataset_params)


def get_dataset_factory(dataset_type):
    if dataset_type == "neph_lm":
        return TextbookDatasetFactory()
    elif dataset_type == "neph_vqa":
        return NephVQADatasetFactory()
    elif dataset_type == "neph_instruct_vqa":
        return NephVQAInstructFinetuneDatasetFactory()
    elif dataset_type == "slake_vqa":
        return SlakeVQADatasetFactory()
    elif dataset_type == "slake_llava_vqa":
        return SlakeLlavaVQADatasetFactory()
    elif dataset_type == "brats_3d_vqa":
        return Brats3DVQADatasetFactory()
    elif dataset_type == "brats_3d2d_vqa":
        return Brats3D2DVQADatasetFactory()
    elif dataset_type == "roccov2_vqa":
        return ROCCOv2VQADatasetFactory()
    elif dataset_type == 'mimic_vqa':
        return MimicVQADatasetFactory()
    elif dataset_type == 'mimic_finetune':
        return MimicFinetuneDatasetFactory()
    elif dataset_type == 'kg_finetune':
        return KGFinetuneDatasetFactory()
    elif dataset_type == "slake_kg":
        return SlakeKGFactory()
    else:
        raise ValueError("Invalid dataset type: {}".format(dataset_type))
