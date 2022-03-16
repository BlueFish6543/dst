import logging
import operator
import pathlib
import sys
import time
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from src.dst.dataset import (
    TrainDataset,
    Vocabulary
)
from src.dst.utils import set_seed, save_checkpoint, load_model, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def get_dataloader(args, tokenizer, filename, sampler, data_size=-1):
    dataset = TrainDataset(args, tokenizer, filename, data_size)
    dataloader = DataLoader(
        dataset,
        sampler=sampler(dataset),
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn
    )
    return dataloader


def score_dev(args, dataloader, model):
    loss_total = 0
    num_batches = 0
    model.eval()
    start_time = time.time()
    for batch in tqdm(dataloader, desc="Dev", disable=args.verbose.disable_display):
        num_batches += 1
        with torch.no_grad():
            output = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=batch['label_ids'].to(DEVICE)
            )
        loss_total += output.loss.mean().item()
    return loss_total / num_batches, time.time() - start_time


def train(args, tokenizer, model, train_dataloader, dev_dataloader,
          optimizer, scheduler, initial_step=0):
    train_dev_args = args
    dev_args, train_args = args.dev, args.train
    log_dir = Path().resolve().joinpath("runs/{}".format(train_args.experiment_name))
    writer = SummaryWriter(
        log_dir=str(log_dir),
    )
    logger.info(f"Tensorboard logs saved at: {log_dir}")
    
    eval_step = dev_args.eval_interval // train_args.batch_size
    gstep = initial_step // train_args.batch_size

    loss_dev, t = score_dev(dev_args, dev_dataloader, model)
    logger.info(f"Epoch: {gstep} | Dev loss: {loss_dev:.8f} | Time: {t:.3f}")
    if gstep > 0:
        # We can't actually read the plot if we log that value
        writer.add_scalar('Loss/dev', loss_dev, global_step=gstep * train_args.batch_size)
    dev_loss_curve = [(loss_dev, t, 0)]
    logger.info('Start training!')

    for epoch in range(train_args.epochs):
        # Initialise for each epoch
        start_time = time.time()
        loss_disp = 0
        model.train()
        model.zero_grad()

        iterator = enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=train_args.verbose.disable_display))
        local_step = 0
        for local_step, batch in iterator:
            output = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=batch['label_ids'].to(DEVICE),
            )
            loss = output.loss.mean()
            loss_disp += output.loss.mean().item()
            gstep += 1
            # Update model
            if loss.item() != 0:
                loss = loss / train_args.gradient_accumulation_steps
                loss.backward()
            if gstep % train_args.gradient_accumulation_steps == 0:
                optimizer.step()
                if train_args.use_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            if gstep % eval_step == 0:
                loss_dev, t = score_dev(dev_args, dev_dataloader, model)
                dev_loss_curve.append((loss_dev, t, gstep))
                model.train()
                logger.info(f"Epoch: {epoch} | Batch: {gstep} | Dev loss: {loss_dev:.8f} | Time: {t:.3f}")
                writer.add_scalar('Loss/dev', loss_dev, global_step=gstep * train_args.batch_size)
                save_checkpoint(train_dev_args, tokenizer, model, gstep * train_args.batch_size,
                                optimizer, scheduler)

        loss_disp /= (local_step + 1)
        logger.info(
            f"Epoch: {epoch} | Batch: {gstep} | Train loss: {loss_disp:.8f} | Time: {time.time() - start_time:.3f}")
        writer.add_scalar('Loss/train', loss_disp, global_step=gstep * train_args.batch_size)
        loss_dev, t = score_dev(dev_args, dev_dataloader, model)
        logger.info(f"Epoch: {epoch} | Batch: {gstep} | Dev loss: {loss_dev:.8f} | time: {t:.3f}")
        writer.add_scalar('Loss/dev', loss_dev, global_step=gstep * train_args.batch_size)

    dev_loss_curve.sort(key=operator.itemgetter(0))
    logger.info(
        f"Lowest dev loss: {dev_loss_curve[0][0]} | Step: {dev_loss_curve[0][2]} | Time: {dev_loss_curve[0][1]}.")


def set_model(args):
    # Initiate config, tokeniser and model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if 'gpt2' in args.model_name_or_path.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
    elif 't5' in args.model_name_or_path.lower():
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    else:
        raise ValueError("Unsupported model.")
    vocabulary = Vocabulary()
    tokenizer.add_special_tokens(vocabulary.special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    # TODO: CONFIGURE THIS
    model = nn.DataParallel(model)
    model.to(DEVICE)
    return config, tokenizer, model


@click.command()
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option(
    "-t",
    "--train-data",
    "train_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to training data.",
)
@click.option(
    "-d",
    "--dev-data",
    "dev_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to dev data.",
)
@click.option(
    "-a",
    "--args",
    "args_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the file with the training arguments.",
)
@click.option(
    "-r",
    "--restore",
    "ckpt_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the checkpoint folder from where the model is to be loaded.",
)
def main(
        args_path: pathlib.Path,
        train_path: pathlib.Path,
        dev_path: pathlib.Path,
        log_level: int,
        ckpt_path: pathlib.Path
):
    args = OmegaConf.load(args_path)
    log_dir = Path(args.train.checkpoint_dir).joinpath(args.train.experiment_name, 'logs').resolve()
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True, parents=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(
            '{}.log'.format(log_dir.joinpath(Path(__file__).stem)),
            mode='a' if ckpt_path else 'w',
        )
    ]
    logging.basicConfig(
        handlers=handlers,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)
    if ckpt_path:
        logger.info(f"Restarting training from checkpoint: {ckpt_path}")

    logger.info(OmegaConf.to_yaml(args))
    logger.info("Training on: {}".format('GPU' if 'cuda' in DEVICE.type else 'CPU'))
    set_seed(args.reproduce)
    args.train.dst_train_path = str(train_path)
    args.dev.dst_dev_path = str(dev_path)

    initial_step = 0 if not ckpt_path else int(ckpt_path.suffix[1:])

    if ckpt_path:
        # Load from checkpoint
        args.train.checkpoint = str(ckpt_path)
        config, tokenizer, model = load_model(args.train, device=DEVICE)
    else:
        config, tokenizer, model = set_model(args.train)

    train_dataloader = get_dataloader(
        args.train,
        tokenizer,
        args.train.dst_train_path,
        sampler=RandomSampler,
        data_size=args.train.data_size
    )
    dev_dataloader = get_dataloader(
        args.dev,
        tokenizer,
        args.dev.dst_dev_path,
        sampler=SequentialSampler,
        data_size=args.dev.data_size
    )
    
    # if 'gpt2' in args.train.model_name_or_path.lower():
    if True:
        optimizer = AdamW(
            model.parameters(),
            lr=args.train.learning_rate,
            eps=args.train.adam_eps
        )
    # elif 't5' in args.train.model_name_or_path.lower():
    #     optimizer = Adafactor(
    #         model.parameters(),
    #         lr=args.train.learning_rate,
    #         scale_parameter=False,
    #         relative_step=False,
    #         warmup_init=False
    #     )
    else:
        raise ValueError("Unsupported model.")
    scheduler = None
    if args.train.use_scheduler:
        t_total = len(train_dataloader) // args.train.gradient_accumulation_steps * args.train.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.train.warmup_steps,
            num_training_steps=t_total
        )
    if ckpt_path:
        optimizer, scheduler = load_checkpoint(ckpt_path, optimizer, scheduler)

    train(args, tokenizer, model, train_dataloader, dev_dataloader,
          optimizer, scheduler, initial_step=initial_step)


if __name__ == '__main__':
    main()


# TODO: CONFIRM_GOOGLE: MULTIPLE VALUE STRATEGY
# TODO: CONFIRM_GOOGLE: OPTIMIZER
# TODO: CONFIRM_GOOGLE: PADDING SIDE
# TODO: CONFIRM_GOOGLE: SPECIAL_TOKENS_HANDLING
# TODO: CONFIRM_GOOGLE: INPUT SEPARATORS
# TODO: CONFIRM_GOOGLE: APPENDIX MISTAKES
