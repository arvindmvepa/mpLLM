from abc import abstractmethod, ABC
import sys


class ModelFactory(ABC):
    """Abstract class for ModelFactory"""

    @abstractmethod
    def create_model(self, exp_file=None):
        raise NotImplementedError()


class LLMTrainerFactory(ModelFactory):
    def create_model(self, exp_file=None):
        from model.llm_trainer import LLMTrainer
        return LLMTrainer(exp_file=exp_file)


class Blip2ProjTrainerFactory(ModelFactory):
    def create_model(self, exp_file=None):
        from model.blip2_proj_trainer import Blip2ProjTrainer
        return Blip2ProjTrainer(exp_file=exp_file)


class ClipProjTrainerFactory(ModelFactory):
    def create_model(self, exp_file=None):
        from model.clip_proj_trainer import ClipProjTrainer
        return ClipProjTrainer(exp_file=exp_file)


class Med3DLLMTrainerFactory(ModelFactory):
    def create_model(self, exp_file=None):
        from model.medical_3D_llm_trainer import Medical3DLLMTrainer
        return Medical3DLLMTrainer(exp_file=exp_file)


class ClipProjTrainerWithRAGFactory(ModelFactory):
    def create_model(self, exp_file=None):
        from model.clip_proj_trainer import ClipProjTrainerWithRAG
        return ClipProjTrainerWithRAG(exp_file=exp_file)


class LlavaMedTrainerFactory(ModelFactory):
    def create_model(self, exp_file):
        sys.path.append("./MedTrinity-25M")
        from model.llava_med_trainer import LlavaMedTrainer
        return LlavaMedTrainer(exp_file)


class OpenaiLLMFactory(ModelFactory):
    def create_model(self, exp_file=None):
        from model.openai_llm import OpenaiLLM
        return OpenaiLLM(exp_file=exp_file)


class KGEmbedderFactory(ModelFactory):
    def create_model(self, exp_file=None):
        from model.kg_embedder import KGEmbedder
        return KGEmbedder(exp_file=exp_file)


def get_model_factory(model_type):
    if model_type == "llm":
        return LLMTrainerFactory()
    elif model_type == "blip2":
        return Blip2ProjTrainerFactory()
    elif model_type == "clip":
        return ClipProjTrainerFactory()
    elif model_type == "clip_w_rag":
        return ClipProjTrainerWithRAGFactory()
    elif model_type == "med_3d_llm":
        return Med3DLLMTrainerFactory()
    elif model_type == "llava_med":
        return LlavaMedTrainerFactory()
    elif model_type == "openai_llm":
        return OpenaiLLMFactory()
    elif model_type == "kg_embed":
        return KGEmbedderFactory()
    else:
        raise ValueError("Invalid model type: {}".format(model_type))



