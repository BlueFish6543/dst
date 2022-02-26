from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Union

import torch
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
        self.eos_id = tokenizer.eos_token_id
        self.ignore_token_id = -100
        self.max_seq_len = args.max_seq_len
        with open(filename, 'r') as f:
            dataset = json.load(f)
            self.data = dataset["data"]
            self.separators = dataset["separators"]
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
    def __init__(self, args, tokenizer, filename, data_size):
        super().__init__(args, tokenizer, filename, data_size)

    def _create_examples(self):
        self.examples = []
        over_length = 0
        skip_counter = 0
        intent_examples = 0
        for dialogue_id, dialogue in tqdm(
                self.data.items(),
                desc=f"Loading {self.filename}\n",
                disable=self.args.verbose.disable_display
        ):
            if self.data_size != -1 and len(self.examples) >= self.data_size:
                break
            context = ""
            for turn_index, turn in enumerate(dialogue):
                user_utterance = turn['user_utterance']
                system_utterance = turn['system_utterance']
                if not system_utterance:
                    utterance = f"<USR> {user_utterance} "
                else:
                    utterance = f"<SYS> {system_utterance} <USR> {user_utterance} "
                context += utterance

                # Intent
                for service in turn['intent_dict']:
                    description = turn['intent_dict'][service]["description"]
                    active = turn['intent_dict'][service]["active"]
                    mapping = turn['intent_dict'][service]["mapping"]
                    # Intent: Service: description 1: intent 2: intent ...
                    # <USR> ... <SYS> ... <USR> ...
                    model_input = description + " " + context
                    context_ids = self.tokenizer(model_input.strip())['input_ids']
                    if active:
                        target_ids = self.tokenizer(str(mapping[active]))['input_ids']
                    else:
                        target_ids = self.tokenizer("")['input_ids']
                    over_length = self.create_ids(
                        dialogue_id, turn_index, context_ids, target_ids, user_utterance, over_length)
                    intent_examples += 1

                # Iterate per slot
                for service in turn['slot_dict']:
                    for slot in turn['slot_dict'][service]:
                        description = turn['slot_dict'][service][slot]["description"]
                        requested = str(turn['slot_dict'][service][slot]["requested"]).lower()
                        value = turn['slot_dict'][service][slot]["value"]
                        mapping = turn['slot_dict'][service][slot]["mapping"]
                        if requested == 'false' and not value:
                            skip_counter += 1
                            if not (skip_counter % 2):
                                # Skip some examples to balance out intent and slot prediction tasks a bit
                                continue
                        if mapping and value:
                            # Get the index of the categorical value
                            value = "dontcare" if value == "dontcare" else str(mapping[value])
                        target = "requested" + self.separators["pair"] + requested + \
                            self.separators["default"] + "value" + self.separators["pair"] + value

                        # Categorical/Non-categorical: Service: description Slot: description
                        # [1: value 2: value ...] <USR> ... <SYS> ... <USR> ...
                        # requested = true/false <SEP> value = value
                        model_input = description + " " + context
                        context_ids = self.tokenizer(model_input.strip())['input_ids']
                        target_ids = self.tokenizer(target.strip())['input_ids']
                        over_length = self.create_ids(
                            dialogue_id, turn_index, context_ids, target_ids, user_utterance, over_length)

        logger.info(f"Data statistics: {self.filename}: {len(self.examples)} examples")
        logger.info(f"Data statistics: {self.filename}: {intent_examples} intent examples")
        logger.info(f"Number of over-length examples: {self.filename}: {over_length} examples")

    def create_ids(self, dialogue_id, turn_index, context_ids, target_ids, user_utterance, over_length):
        target_len = len(target_ids)
        if 'gpt2' in self.args.model_name_or_path.lower():
            # context <BOS> target <EOS>
            input_ids = context_ids + [self.tokenizer.bos_token_id] + target_ids + [self.tokenizer.eos_token_id]
            pad_len = len(input_ids) - target_len - 1  # EOS token
            label_ids = [self.ignore_token_id] * pad_len + target_ids + [self.tokenizer.eos_token_id]
            assert len(input_ids) == len(label_ids)
        elif 't5' in self.args.model_name_or_path.lower():
            input_ids = context_ids
            label_ids = target_ids
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
            'label_ids': label_ids,
            'user_utterance': user_utterance,  # useful for results analysis
            'example_id': f"{dialogue_id}_{turn_index}",
        })
        return over_length

    def collate_fn(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = self._pad(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        label_ids = [example['label_ids'] for example in batch]
        label_ids, _ = self._pad(label_ids, self.ignore_token_id)
        label_ids = torch.tensor(label_ids).long()
        user_utterances = [example['user_utterance'] for example in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids,
            'user_utterance': user_utterances
        }


class TestDataset(DSTDataset):
    def __init__(self, args, tokenizer, filename, data_size):
        self.to_decode: set[str] = set(args.decode_only)
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
                    utterance = f"<USR> {user_utterance} "
                else:
                    utterance = f"<SYS> {system_utterance} <USR> {user_utterance} "
                context += utterance

                # Intent
                for service in turn['intent_dict']:
                    description = turn['intent_dict'][service]["description"]
                    # Intent: Service: description 1: intent 2: intent ...
                    # <USR> ... <SYS> ... <USR> ...
                    model_input = description + " " + context
                    context_ids = self.tokenizer(model_input.strip())['input_ids']
                    over_length = self.create_ids(
                        dialogue_id, turn_index, context_ids, user_utterance, over_length, service=service)

                # Iterate per slot
                for service in turn['slot_dict']:
                    for slot in turn['slot_dict'][service]:
                        description = turn['slot_dict'][service][slot]["description"]
                        # Categorical/Non-categorical: Service: description Slot: description
                        # [1: value 2: value ...] <USR> ... <SYS> ... <USR> ...
                        # requested = true/false <SEP> value = value
                        model_input = description + " " + context
                        context_ids = self.tokenizer(model_input.strip())['input_ids']
                        over_length = self.create_ids(
                            dialogue_id, turn_index, context_ids, user_utterance, over_length,
                            service=service, slot=slot)

        logger.info(f"Data statistics: {self.filename}: {len(self.examples)} examples")
        logger.info(f"Number of over-length examples: {self.filename}: {over_length} examples")

    def create_ids(self, dialogue_id, turn_index, context_ids, user_utterance, over_length,
                   service=None, slot=None):
        if 'gpt2' in self.args.model_name_or_path.lower():
            # context <BOS> target <EOS>
            dst_input_ids = context_ids + [self.tokenizer.bos_token_id]
        elif 't5' in self.args.model_name_or_path.lower():
            dst_input_ids = context_ids
        else:
            raise ValueError("Unsupported model.")
        if len(dst_input_ids) > self.max_seq_len:
            over_length += 1
            dst_input_ids = dst_input_ids[-self.max_seq_len:]
        self.examples.append({
            'input_ids': dst_input_ids,
            'example_id': f"{dialogue_id}_{turn_index}",
            'user_utterance': user_utterance,
            'service': service,
            'slot': slot
        })
        return over_length

    def collate_fn(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = self._pad(input_ids, self.pad_id)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        example_id = [example['example_id'] for example in batch]
        user_utterances = [example['user_utterance'] for example in batch]
        services = [example['service'] for example in batch]
        slots = [example['slot'] for example in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'example_id': example_id,
            'user_utterance': user_utterances,
            'service': services,
            'slot': slots
        }


if __name__ == '__main__':
    pass
