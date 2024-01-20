from typing import Callable, Tuple
from abc import ABCMeta
from . import attention_unet3d_diffusion, resnet_2d, resnet_3d, sparse_unet3d, transformer, unet_2d, unet_3d
import timm
from transformers import AutoModel
from diffusers import AutoencoderKL

# TODO: better designing the model interface


def _is_available_on_timm(model_id: str) -> bool:
    try:
        return len(timm.list_models(model_id)) > 0
    except Exception:
        return False


def _create_timm_model(model_id: str, **kwargs):
    return timm.create_model(model_id, **kwargs)


def _is_available_on_huggingface(model_id: str) -> bool:
    try:
        return AutoModel.from_pretrained(model_id)
    except Exception:
        return False


def _is_available_on_diffuser(model_id: str) -> bool:
    try:
        return AutoencoderKL.from_pretrained(model_id)
    except Exception:
        return False


def _create_diffuser_model(model_id: str, **kwargs):
    return AutoencoderKL.from_pretrained(model_id, **kwargs)


def _create_huggingface_model(model_id: str, **kwargs):
    return AutoModel.from_pretrained(model_id, **kwargs)


def create_model(model_id: str, **kwargs):
    if model_id in getattr(resnet_3d, "__all__"):
        return getattr(resnet_3d, model_id)(**kwargs)
    elif model_id in getattr(resnet_2d, "__all__"):
        return getattr(resnet_2d, model_id)(**kwargs)
    elif model_id in getattr(sparse_unet3d, "__all__"):
        return getattr(sparse_unet3d, model_id)(**kwargs)
    elif model_id in getattr(unet_3d, "__all__"):
        return getattr(unet_3d, model_id)(**kwargs)
    elif model_id in getattr(unet_2d, "__all__"):
        return getattr(unet_2d, model_id)(**kwargs)
    elif model_id in getattr(transformer, "__all__"):
        return getattr(transformer, model_id)(**kwargs)
    elif model_id in getattr(attention_unet3d_diffusion, "__all__"):
        return getattr(attention_unet3d_diffusion, model_id)(**kwargs)
    elif _is_available_on_timm(model_id):
        return _create_timm_model(model_id, **kwargs)
    elif _is_available_on_huggingface(model_id):
        return _create_huggingface_model(model_id, **kwargs)
    elif _is_available_on_diffuser(model_id):
        return _create_diffuser_model(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown model id: {model_id}")
