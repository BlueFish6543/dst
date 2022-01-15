from __future__ import annotations

import json
import logging
import torch

from dataclasses import dataclass, field
from typing import Union
from tqdm import tqdm

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    "sep_token": "<SEP>",
    "additional_special_tokens": ["<USR>", "<SYS>"]
}


@dataclass
class Vocabulary:
    special_tokens: dict[str, Union[str, list[str]]] = field(default_factory=lambda: SPECIAL_TOKENS)
    vocabulary_update: bool = False

    def add_special_tokens(self, tokens: dict[str, str]):
        for token in tokens:
            token = token.upper().strip()
            for value in self.special_tokens.values():
                if token in value:
                    break
            else:
                self.special_tokens["additional_special_tokens"].append(token)
        if tokens:
            self.vocabulary_update = True


class DSTDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, filename, data_size):
        self.args = args
        self.data_size = data_size
        self.tokenizer = tokenizer
        self.filename = filename
        self.pad_id = tokenizer.pad_token_id  # pad to max example length
        self.ignore_token_id = -100
        self.max_seq_len = args.max_seq_len
        with open(filename, 'r') as f:
            dataset = json.load(f)
            self.data = dataset["data"]
        self._create_examples()

    @staticmethod
    def _pad(sentences, pad_id, side="right"):
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

    def __len__(self):  # required
        return len(self.examples)

    def __getitem__(self, index):  # required
        return self.examples[index]


class TrainDataset(DSTDataset):
    def __init__(self, args, tokenizer, filename, data_size, add_slot_names=False):
        self.add_slot_names = add_slot_names
        super().__init__(args, tokenizer, filename, data_size)

    def _create_examples(self):
        self.examples = []
        over_length = 0
        for dialogue_id, dialogue in tqdm(
                self.data.items(),
                desc=f"Loading {self.filename}\n",
                disable=self.args.verbose.disable_display
        ):
            if self.data_size != -1 and len(self.examples) >= self.data_size:
                break
            context = ""
            for turn_index, turn in enumerate(dialogue):
                slot_values = turn['slot_values']
                user_utterance = turn['user_utterance']
                system_utterance = turn['system_utterance']

                if 'gpt2' in self.args.model_name_or_path.lower():
                    if not system_utterance:
                        if self.add_slot_names:
                            utterance = turn['slot_names'] + f" <USR> {user_utterance} "
                        else:
                            utterance = f"<USR> {user_utterance} "
                    else:
                        utterance = f"<SYS> {system_utterance} <USR> {user_utterance} "
                    context += utterance
                    context_ids = self.tokenizer(context)['input_ids']
                    target_ids = self.tokenizer(slot_values)['input_ids']
                    target_len = len(target_ids)

                    # context <BOS> target <EOS>
                    input_ids = context_ids + [self.tokenizer.bos_token_id] + target_ids + [self.tokenizer.eos_token_id]
                    pad_len = len(input_ids) - target_len - 1  # EOS token
                    label_ids = [self.ignore_token_id] * pad_len + target_ids + [self.tokenizer.eos_token_id]
                    assert len(input_ids) == len(label_ids)
                    decoder_ids = None

                elif 't5' in self.args.model_name_or_path.lower():
                    if not system_utterance:
                        utterance = f"<USR> {user_utterance} "
                    else:
                        utterance = f"<SYS> {system_utterance} <USR> {user_utterance} "
                    context += utterance
                    context_ids = self.tokenizer(context)['input_ids']
                    decoder_ids = self.tokenizer(turn['slot_names'])['input_ids'][:-1]  # discard last end token
                    target_ids = self.tokenizer(slot_values)['input_ids']
                    target_len = len(target_ids)

                    input_ids = context_ids
                    decoder_ids += [self.tokenizer.pad_token_id] + target_ids  # T5 uses pad token as start token
                    pad_len = len(decoder_ids) - target_len
                    label_ids = [self.ignore_token_id] * pad_len + target_ids
                    assert len(decoder_ids) == len(label_ids)

                else:
                    raise ValueError("Unsupported model.")

                if len(input_ids) > self.max_seq_len:
                    # Handle over-length example
                    logger.warning(f"{dialogue_id}({turn_index}) exceeds maximum sequence length, truncating...")
                    over_length += 1
                    input_ids = input_ids[-self.max_seq_len:]
                    label_ids = label_ids[-self.max_seq_len:]
                assert len(input_ids) <= self.max_seq_len

                self.examples.append({
                    'input_ids': input_ids,
                    'decoder_ids': decoder_ids,
                    'label_ids': label_ids,
                    'user_utterance': user_utterance,  # useful for results analysis
                    'example_id': f"{dialogue_id}_{turn_index}",
                })
        logger.info(f"Data statistics: {self.filename}: {len(self.examples)} examples")
        logger.info(f"Number of over-length examples: {self.filename}: {over_length} examples")

    def collate_fn(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = self._pad(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        if 'gpt2' in self.args.model_name_or_path.lower():
            decoder_ids = None
            label_ids = [example['label_ids'] for example in batch]
            label_ids, _ = self._pad(label_ids, self.ignore_token_id)
            label_ids = torch.tensor(label_ids).long()
        elif 't5' in self.args.model_name_or_path.lower():
            decoder_ids = [example['decoder_ids'] for example in batch]
            decoder_ids, _ = self._pad(decoder_ids, self.pad_id, side="left")
            decoder_ids = torch.tensor(decoder_ids).long()
            label_ids = [example['label_ids'] for example in batch]
            label_ids, _ = self._pad(label_ids, self.ignore_token_id, side="left")
            label_ids = torch.tensor(label_ids).long()
        else:
            raise ValueError("Unknown model.")
        user_utterances = [example['user_utterance'] for example in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_ids': decoder_ids,
            'label_ids': label_ids,
            'user_utterance': user_utterances
        }


class TestDataset(DSTDataset):
    def __init__(self, args, tokenizer, filename, data_size, add_slot_names=False):
        self.to_decode: set[str] = set(args.decode_only)
        self.add_slot_names = add_slot_names
        super().__init__(args, tokenizer, filename, data_size)

    def _create_examples(self):
        self.examples = []
        over_length = 0
        for dialogue_id, dialogue in tqdm(
                self.data.items(),
                desc=f"Loading {self.filename}",
                disable=self.args.verbose.disable_display
        ):
            if self.to_decode and dialogue_id not in self.to_decode:
                continue
            if self.data_size != -1 and len(self.examples) >= self.data_size:
                break
            context = ""

            for turn_index, turn in enumerate(dialogue):
                user_utterance = turn['user_utterance']
                system_utterance = turn['system_utterance']
                if not system_utterance:
                    if self.add_slot_names:
                        utterance = turn['slot_names'] + f" <USR> {user_utterance} "
                    else:
                        utterance = f"<USR> {user_utterance} "
                else:
                    utterance = f"<SYS> {system_utterance} <USR> {user_utterance} "
                context += utterance
                dst_input_ids = self.tokenizer(context)['input_ids'] + [self.tokenizer.bos_token_id]
                if len(dst_input_ids) > self.max_seq_len:
                    over_length += 1
                    dst_input_ids = dst_input_ids[-self.max_seq_len:]

                self.examples.append({
                    'input_ids': dst_input_ids,
                    'example_id': f"{dialogue_id}_{turn_index}",
                    'user_utterance': user_utterance,
                })
        logger.info(f"Data statistics: {self.filename}: {len(self.examples)} examples")
        logger.info(f"Number of over-length examples: {self.filename}: {over_length} examples")

    def collate_fn(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = self._pad(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        example_id = [example['example_id'] for example in batch]
        user_utterances = [example['user_utterance'] for example in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'example_id': example_id,
            'user_utterance': user_utterances
        }


if __name__ == '__main__':
    pass
