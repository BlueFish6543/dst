import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    # For reproduction
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn.enabled
    torch.backends.cudnn.enabled = args.cudnn.deterministic
    torch.backends.cudnn.benchmark = args.cudnn.benchmark


def save_checkpoint(args, tokenizer, model, step, optimizer, scheduler):
    ckpt_path = Path(args.train.checkpoint_dir)
    ckpt_path = ckpt_path.joinpath(args.train.experiment_name)
    if not ckpt_path.exists():
        ckpt_path.mkdir(exist_ok=True, parents=True)
    save_path = f"{ckpt_path}/model.{step}"
    logger.info(f"Save model in {save_path}!")
    tokenizer.save_pretrained(save_path)
    model.module.save_pretrained(save_path)
    OmegaConf.save(args, f"{ckpt_path}/model_config.yaml")
    state_dict = {'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        state_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(save_path, "checkpoint.pth"))


def load_model(args, device: torch.device):
    ckpt_path = args.checkpoint
    logger.info(f"Load model, tokenizer from {ckpt_path}")
    if 'gpt2' in args.model_name_or_path.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path)
        model = GPT2LMHeadModel.from_pretrained(ckpt_path)
    elif 't5' in args.model_name_or_path.lower():
        tokenizer = T5Tokenizer.from_pretrained(ckpt_path)
        model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
    else:
        raise ValueError("Unsupported model.")
    model = nn.DataParallel(model)
    model.to(device)
    return model.module.config, tokenizer, model


def load_checkpoint(ckpt_path, optimizer, scheduler):
    checkpoint = torch.load(os.path.join(ckpt_path, "checkpoint.pth"))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return optimizer, scheduler


def humanise(
        name: str,
        remove_trailing_numbers: bool = False
) -> str:
    # Convert a potentially camel or snake case string to a lower case string delimited by spaces
    # Adapted from https://stackoverflow.com/a/1176023
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    name = name.lower().replace("_", " ")
    if remove_trailing_numbers:
        # Remove trailing numbers
        name = re.sub('[0-9]+$', '', name)
    return re.sub(' +', ' ', name).strip()
