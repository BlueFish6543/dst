from __future__ import annotations

import collections
import json
import logging
import pathlib
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
import transformers
from omegaconf import DictConfig
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from dst.utils import (
    PathMapping,
    Schema,
    infer_schema_variant_from_path,
    infer_split_name_from_path,
    load_schema,
)

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    "sep_token": "<SEP>",
    "additional_special_tokens": [],
}


def reconstruct_filename(dial_id: str) -> str:
    """Reconstruct filename from dialogue ID."""

    file_prefix = int(dial_id.split("_")[0])

    if file_prefix in range(10):
        str_file_prefix = f"00{file_prefix}"
    elif file_prefix in range(10, 100):
        str_file_prefix = f"0{file_prefix}"
    else:
        str_file_prefix = f"{file_prefix}"

    return f"dialogues_{str_file_prefix}.json"


def get_file_map(
    dialogue_ids: list[str],
    split: Literal["train", "test", "dev"],
    data_pckg_or_path: str = "data.raw",
) -> dict[pathlib.Path, list[str]]:
    """Returns a map where the keys are file paths and values are lists
    comprising dialogues from `dialogue_ids` that are in the same file.

    dialogue_ids:
        IDs of the dialogues whose paths are to be returned, formated as the schema 'dialogue_id' field.
    split:
        The name of the split whose paths are to be returned.
    data_pckg_or_path:
        The location of the python package where the data is located
    """

    file_map = collections.defaultdict(list)
    path_map = PathMapping(data_pckg_or_path=data_pckg_or_path)
    for d_id in dialogue_ids:
        # ValueError occurs if dialogue IDs do not match SGD convention
        try:
            fpath = path_map[split].joinpath(reconstruct_filename(d_id))
        except ValueError:
            found_dialogue = False
        else:
            # for the original SGD data, one can reconstruct the filename
            # from dial ID to load the dialogue
            file_map[fpath].append(d_id)
            continue
        # in general, just iterate through the file to find a given
        # dialogue
        if not found_dialogue:
            for fpath in path_map[split].iterdir():
                if not fpath.name.startswith("dialogues"):
                    continue
                with open(fpath, "r") as f:
                    dial_bunch = json.load(f)
                for dial in dial_bunch:
                    if dial["dialogue_id"] == d_id:
                        found_dialogue = True
                        break

                if found_dialogue:
                    break

            if found_dialogue:
                file_map[fpath].append(d_id)
            else:
                logging.warning(f"Could not find dialogue {d_id}...")

    return file_map


@dataclass
class Vocabulary:
    special_tokens: dict[str, Union[str, list[str]]] = field(
        default_factory=lambda: SPECIAL_TOKENS
    )
    vocabulary_update: bool = False

    def add_special_tokens(self, tokens: dict[str, str]):
        for token in tokens:
            token = token.strip()
            for value in self.special_tokens.values():
                if token in value:
                    break
            else:
                self.special_tokens["additional_special_tokens"].append(token)
        if tokens:
            self.vocabulary_update = True


def get_inference_data_loader(args, tokenizer):
    dataset = BatchedTestDataset(args, tokenizer, args.dst_test_path, args.data_size)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=None, collate_fn=dataset.collate_fn
    )
    return data_loader


def get_dataloader(
    args: DictConfig,
    tokenizer: transformers.T5Tokenizer,
    data_paths: list[str],
    schema: Schema,
    sampler,
    data_size: int = -1,
) -> DataLoader:
    dataset = TrainDataset(args, tokenizer, data_paths, data_size, schema)
    dataloader = DataLoader(
        dataset,
        sampler=sampler(dataset),
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
    )
    return dataloader


def pad(sentences, pad_id, side="right"):
    max_len = max((map(len, sentences)))
    attention_mask = []
    sentences_pad = []
    for sent in sentences:
        pad_len = max_len - len(sent)
        if side == "right":
            sentences_pad.append(sent + [pad_id] * pad_len)
            attention_mask.append([1] * len(sent) + [0] * pad_len)
        elif side == "left":
            sentences_pad.append([pad_id] * pad_len + sent)
            attention_mask.append([0] * pad_len + [1] * len(sent))
        else:
            raise ValueError("Unknown padding side.")
    return sentences_pad, attention_mask


class DSTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args: DictConfig,
        tokenizer,
        data_paths: list[str],
        data_size: int,
        train_schema: Schema,
    ):
        self.args = args
        self.data_size = data_size
        self.tokenizer = tokenizer
        self.data_paths = data_paths
        self.pad_id = tokenizer.pad_token_id  # pad to max example length
        self.ignore_token_id = -100
        self.max_seq_len = args.max_seq_len
        self.decoder_max_seq_len = args.decoder_max_seq_len
        self.examples = []
        self.seen_services = set(train_schema.services)  # type: set[str]
        inferred_splits = {infer_split_name_from_path(pth) for pth in data_paths}
        assert (
            len(inferred_splits) == 1
        ), f"Attempting to load data from multiple splits: _{inferred_splits}_"
        self.split = list(inferred_splits)[0]
        inferred_schema_variants = [
            infer_schema_variant_from_path(pth) for pth in data_paths
        ]
        if len(inferred_schema_variants) != len(set(inferred_schema_variants)):
            duplicates = collections.Counter(inferred_schema_variants)
            raise AssertionError(
                "When loading data, encountered multiple shards from same sgd-x variant: \n"
                f"{duplicates.most_common()}"
            )
        self.schema_variants = inferred_schema_variants
        self._create_examples()
        self.dialogue_files = None
        self.to_decode = set()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def _create_examples(self):
        raise NotImplementedError

    def _get_dialogue_ids(self):
        dialogue_ids = set()
        for example in self.examples:
            if isinstance(example, list):
                dialogue_ids.update([e["dialogue_id"] for e in self.examples])
            else:
                assert isinstance(example, dict)
                dialogue_ids.add(example["dialogue_id"])
        return dialogue_ids

    def _infer_dialogue_files(self):
        if self.data_size == -1 and not self.to_decode:
            return
        try:
            data_pckg_or_path = self.args.ref_path
        except AttributeError:
            return
        dialogue_ids = self._get_dialogue_ids()
        self.dialogue_files = [
            p.name
            for p in get_file_map(
                list(dialogue_ids), self.split, data_pckg_or_path=data_pckg_or_path
            )
        ]


class TrainDataset(DSTDataset):
    def __init__(
        self,
        args: DictConfig,
        tokenizer,
        data_paths: list[str],
        data_size: int,
        train_schema: Schema,
    ):
        self.encoder_over_length = 0
        self.decoder_over_length = 0
        super().__init__(args, tokenizer, data_paths, data_size, train_schema)

    def _create_examples(self):

        for shard_path in self.data_paths:
            logger.info(f"Loading data from shard {shard_path}")
            schema_variant = infer_schema_variant_from_path(shard_path)
            with open(shard_path, "r") as f:
                this_shard_data = json.load(f)
            shuffled_keys = list(this_shard_data.keys())
            random.shuffle(shuffled_keys)
            for i, dialogue_id in enumerate(shuffled_keys):
                dialogue = this_shard_data[dialogue_id]
                if self.data_size != -1 and i > self.data_size:
                    break
                context = ""
                for turn_index, turn in enumerate(dialogue):
                    user_utterance = turn["user_utterance"]
                    system_utterance = turn["system_utterance"]
                    if not system_utterance:
                        utterance = f"[user] {user_utterance} "
                    else:
                        utterance = (
                            f"[system] {system_utterance} [user] {user_utterance} "
                        )
                    context += utterance
                    frames = turn["frames"]
                    for service in frames:
                        model_input = frames[service]["description"] + " " + context
                        context_ids = self.tokenizer(model_input)["input_ids"]
                        target_ids = self.tokenizer(frames[service]["expected_output"])[
                            "input_ids"
                        ]
                        self.create_ids(
                            context_ids,
                            target_ids,
                            user_utterance,
                            schema_variant,
                            service,
                            dialogue_id,
                            turn_index,
                        )

        logger.info(
            f"Data statistics: {self.data_paths}: {len(self.examples)} examples"
        )
        logger.info(
            f"Number of input over-length examples: {self.data_paths}: {self.encoder_over_length} examples"
        )
        logger.info(
            f"Number of output over-length examples: {self.data_paths}: {self.decoder_over_length} examples"
        )
        if self.data_size != -1:
            random.shuffle(self.examples)
            self.examples = self.examples[: self.data_size]

    def create_ids(
        self,
        context_ids: list[int],
        target_ids: list[int],
        user_utterance: str,
        schema_variant_identifier: str,
        service: str,
        dialogue_id: str,
        turn_index: int,
    ):
        input_ids = context_ids
        label_ids = target_ids
        if len(input_ids) > self.max_seq_len:
            # Handle over-length example
            logger.warning(
                f"{dialogue_id}({turn_index}) input exceeds maximum sequence length, truncating..."
            )
            self.encoder_over_length += 1
            input_ids = input_ids[-self.max_seq_len :]
        if len(label_ids) > self.decoder_max_seq_len:
            logger.warning(
                f"{dialogue_id}({turn_index}) output exceeds maximum sequence length, truncating..."
            )
            label_ids = label_ids[-self.decoder_max_seq_len :]
        assert len(input_ids) <= self.max_seq_len
        seen_flag = 1
        if self.split != "train":
            if schema_variant_identifier == "original":
                seen_flag = 1 if service in self.seen_services else 0
            else:
                seen_flag = 1 if service[:-1] in self.seen_services else 0
        self.examples.append(
            {
                "input_ids": input_ids,
                "label_ids": label_ids,
                "seen": seen_flag,
                "user_utterance": user_utterance,  # useful for results analysis
                "example_id": f"{schema_variant_identifier}_{dialogue_id}_{turn_index}",
                "dialogue_id": dialogue_id,
            }
        )

    def collate_fn(self, batch):
        input_ids = [example["input_ids"] for example in batch]
        input_ids, attention_mask = pad(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        label_ids = [example["label_ids"] for example in batch]
        label_ids, _ = pad(label_ids, self.ignore_token_id)
        label_ids = torch.tensor(label_ids).long()
        user_utterances = [example["user_utterance"] for example in batch]
        seen_flag = [example["seen"] for example in batch]
        seen_flag = torch.tensor(seen_flag, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "user_utterance": user_utterances,
            "seen": seen_flag,
        }


class TestDataset(DSTDataset):
    def __init__(
        self,
        args: DictConfig,
        tokenizer,
        data_paths: list[str],
        data_size: int,
        train_schema: Schema,
    ):
        self.to_decode: set[str] = set(args.decode_only)
        self.encoder_over_length = 0
        super().__init__(args, tokenizer, data_paths, data_size, train_schema)

    def _create_examples(self):
        assert (
            len(self.data_paths) == 1
        ), "Only one schema variant can be decoded at one time"
        self.examples = []
        with open(self.data_paths[0], "r") as f:
            data = json.load(f)
        schema_variant_identifier = infer_schema_variant_from_path(self.data_paths[0])
        for dialogue_id, dialogue in tqdm(
            data.items(),
            desc=f"Loading {self.data_paths[0]}",
            disable=self.args.verbose.disable_display,
        ):

            if self.to_decode and dialogue_id not in self.to_decode:
                continue
            if self.data_size != -1 and len(self.examples) >= self.data_size:
                break
            context = ""
            for turn_index, turn in enumerate(dialogue):
                user_utterance = turn["user_utterance"]
                system_utterance = turn["system_utterance"]
                if not system_utterance:
                    utterance = f"[user] {user_utterance} "
                else:
                    utterance = f"[system] {system_utterance} [user] {user_utterance} "
                context += utterance
                frames = turn["frames"]
                for service in frames:
                    model_input = frames[service]["description"] + " " + context
                    context_ids = self.tokenizer(model_input)["input_ids"]
                    self.create_ids(
                        context_ids,
                        user_utterance,
                        schema_variant_identifier,
                        service,
                        dialogue_id,
                        turn_index,
                    )

        logger.info(
            f"Data statistics: {self.data_paths}: {len(self.examples)} examples"
        )
        logger.info(
            f"Number of input over-length examples: {self.data_paths}: {self.encoder_over_length} examples"
        )

    def create_ids(
        self,
        context_ids: list[int],
        user_utterance: str,
        schema_variant_identifier: str,
        service: str,
        dialogue_id: str,
        turn_index: int,
    ):
        if "gpt2" in self.args.model_name_or_path.lower():
            # context <BOS> target <EOS>
            dst_input_ids = context_ids + [self.tokenizer.bos_token_id]
        elif "t5" in self.args.model_name_or_path.lower():
            dst_input_ids = context_ids
        else:
            raise ValueError("Unsupported model.")
        if len(dst_input_ids) > self.max_seq_len:
            self.encoder_over_length += 1
            dst_input_ids = dst_input_ids[-self.max_seq_len :]
        if schema_variant_identifier == "original":
            seen_flag = 1 if service in self.seen_services else 0
        else:
            seen_flag = 1 if service[:-1] in self.seen_services else 0
        self.examples.append(
            {
                "input_ids": dst_input_ids,
                "example_id": f"{schema_variant_identifier}_{dialogue_id}_{turn_index}",
                "user_utterance": user_utterance,
                "service": service,
                "seen": seen_flag,
                "dialogue_id": dialogue_id,
            }
        )

    def collate_fn(self, batch):
        input_ids = [example["input_ids"] for example in batch]
        input_ids, attention_mask = pad(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        example_id = [example["example_id"] for example in batch]
        user_utterances = [example["user_utterance"] for example in batch]
        services = [example["service"] for example in batch]
        seen_flag = [example["seen"] for example in batch]
        seen_flag = torch.tensor(seen_flag, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "example_id": example_id,
            "user_utterance": user_utterances,
            "service": services,
            "seen": seen_flag,
        }


class BatchedTestDataset(DSTDataset):
    def __init__(
        self,
        args: DictConfig,
        tokenizer,
        data_paths: list[str],
        data_size: int,
        train_schema: Optional[Schema] = None,
    ):
        self.to_decode: set[str] = set(args.decode_only)
        self.encoder_over_length = 0
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        self.batch_size = args.batch_size
        if train_schema is None:
            train_schema = load_schema(args.orig_train_schema_path)
        # TODO: VERY BAD DESIGN, NEEDS REFACTORING
        super().__init__(args, tokenizer, data_paths, data_size, train_schema)
        self._get_dialogue_ids()

    def _create_examples(self):
        assert (
            len(self.data_paths) == 1
        ), "Only one schema variant can be decoded at one time"
        self.examples = []
        with open(self.data_paths[0], "r") as f:
            data = json.load(f)
        schema_variant_identifier = infer_schema_variant_from_path(self.data_paths[0])
        remaining_examples = self.batch_size
        this_batch_template = {
            "text_input": [],
            "example_id": [],
            "user_utterance": [],
            "service": [],
            "seen": [],
        }
        this_batch = deepcopy(this_batch_template)
        n_frames = 0
        for dialogue_id, dialogue in tqdm(
            data.items(),
            desc=f"Loading {self.data_paths[0]}",
            disable=self.args.verbose.disable_display,
        ):

            if self.to_decode and dialogue_id not in self.to_decode:
                continue
            if self.data_size != -1 and len(self.examples) >= self.data_size:
                break
            context = ""
            for turn_index, turn in enumerate(dialogue):
                user_utterance = turn["user_utterance"]
                system_utterance = turn["system_utterance"]
                if not system_utterance:
                    utterance = f"[user] {user_utterance} "
                else:
                    utterance = f"[system] {system_utterance} [user] {user_utterance} "
                context += utterance
                frames = turn["frames"]
                for service in frames:
                    n_frames += 1
                    encoder_input_text = f"{frames[service]['description']} {context}"
                    if schema_variant_identifier == "original":
                        seen_flag = 1 if service in self.seen_services else 0
                    else:
                        seen_flag = 1 if service[:-1] in self.seen_services else 0
                    this_batch["text_input"].append(encoder_input_text)
                    this_batch["example_id"].append(
                        f"{schema_variant_identifier}_{dialogue_id}_{turn_index}"
                    )
                    this_batch["user_utterance"].append(user_utterance)
                    this_batch["service"].append(service)
                    this_batch["seen"].append(seen_flag)
                    this_batch["dialogue_id"].append(dialogue_id)
                    remaining_examples -= 1
                    if remaining_examples == 0:
                        self.examples.append(self.create_ids(this_batch))
                        remaining_examples = self.batch_size
                        this_batch = deepcopy(this_batch_template)
        if remaining_examples in range(1, self.batch_size):
            assert this_batch["text_input"]
            last_batch = self.create_ids(this_batch)
            self.examples.append(last_batch)
        self._sanity_check(n_frames)
        logger.info(
            f"Data statistics: {self.data_paths[0]}: {len(self.examples)} batches with size {self.batch_size}"
        )

    def create_ids(self, batch: dict):
        model_inputs = self.tokenizer(
            batch["text_input"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        batch["input_ids"] = model_inputs["input_ids"]
        batch["attention_mask"] = model_inputs["attention_mask"]
        return batch

    def collate_fn(self, batch):
        return batch

    def _sanity_check(self, expected_examples: int):
        actual_examples = 0
        for batch in self.examples:
            actual_examples += len(batch["text_input"])
        assert actual_examples == expected_examples


if __name__ == "__main__":
    pass
