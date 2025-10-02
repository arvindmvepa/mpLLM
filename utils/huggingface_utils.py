import torch
import torchvision
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Blip2VisionModel,
    Blip2VisionConfig,
    Blip2QFormerModel,
    Blip2QFormerConfig,
    Blip2Processor
)
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from open_clip import create_model_from_pretrained


def load_tokenizer_from_huggingface(tokenizer_name="HuggingFaceH4/zephyr-7b-beta"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ValueError:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_llm_from_huggingface(model_name="HuggingFaceH4/zephyr-7b-beta", use_quantization=False, r=16, lora_alpha=32,
                              target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                              lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"):
    if "llava" in model_name:
        from llava.model import LlavaLlamaForCausalLM
        model_class = LlavaLlamaForCausalLM
    else:
        model_class = AutoModelForCausalLM
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        model = model_class.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
    else:
        model = model_class.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True)
    model.gradient_checkpointing_enable()

    if r:
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
        )
        model = get_peft_model(model, config)
    return model


def load_model_from_huggingface(model_name="michiyasunaga/BioLinkBERT-base"):
    model = AutoModel.from_pretrained(model_name)
    return model


def load_clip_vision_model(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    model, _ = create_model_from_pretrained(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return model.visual


def load_clip_vision_model_norm_params(model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
    _, preprocess = create_model_from_pretrained(model_name)
    normalize = None
    for t in preprocess.transforms:
        if isinstance(t, torchvision.transforms.transforms.Normalize):
            normalize = t
            break
    return normalize.mean, normalize.std


def load_blip_qformer_model_from_huggingface(model_name="Salesforce/blip2-opt-2.7b"):
    model = Blip2QFormerModel(Blip2QFormerConfig.from_pretrained(model_name))
    return model


def load_blip_vision_model_from_huggingface(model_name="Salesforce/blip2-opt-2.7b"):
    model = Blip2VisionModel(Blip2VisionConfig.from_pretrained(model_name))
    return model


def load_blip_vision_model_norm_params_from_huggingface(model_name="Salesforce/blip2-opt-2.7b"):
    processor = Blip2Processor.from_pretrained(model_name)
    return processor.image_processor.image_mean, processor.image_processor.image_std

def convert_meta_to_tensor(state_dict, device='cpu'):
    for key, param in state_dict.items():
        if param.is_meta:
            # Replace meta tensor with an actual tensor on the specified device
            state_dict[key] = torch.zeros_like(param, device=device)
    return state_dict
