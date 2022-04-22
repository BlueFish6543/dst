from collections import defaultdict

import torch
from tqdm import tqdm


def remove_padding(output_strings: list[str], pad_token: str) -> list(str):
    """When doing batched decoding, shorter sequences are padded. This
    function removes the padding from the output sequences.

    Note:
    ----
    The addition of `pad_token` at the end of each sequence is specific to the T5
    model where the pad symbol is <EOS>. The T5 parser requires an <EOS> symbol at the
    end of the sequence, which is why it is added to every string.
    """
    padding_free = []
    for s in output_strings:
        pad_token_start = s.find(pad_token)
        while pad_token_start != -1:
            s = f"{s[:pad_token_start]}{s[pad_token_start+len(pad_token):].lstrip()}"
            pad_token_start = s.find(pad_token)
        padding_free.append(f"{s} {pad_token}")
    return padding_free


def decode(args, batch, model, tokenizer, device) -> list[str]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if args.generate_api == "huggingface":
        output_seqs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=args.decoder_max_seq_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    else:
        raise ValueError(
            f"Unknown generation API: {args.generate_API}. Only `huggingface' option is valid for batched decoding."
        )
    output_strings = tokenizer.batch_decode(output_seqs)
    return remove_padding(output_strings, tokenizer.pad_token)


def run_inference(args, tokenizer, model, data_loader, device):
    model.eval()
    collector = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    description = (
        f"Decoding {len(data_loader.dataset)} batches from split {data_loader.dataset.split}. "
        f"Schema variants: {data_loader.dataset.schema_variants}"
    )
    with torch.no_grad():
        iterator = enumerate(
            tqdm(
                data_loader,
                desc=description,
                disable=args.verbose.disable_display,
            )
        )
        for step, batch in iterator:
            bs_pred_str_batch = decode(args, batch, model, tokenizer, device)
            for i, bs_pred_str in enumerate(bs_pred_str_batch):
                schema_variant, dial_turn = batch["example_id"][i].split("_", 1)
                dialogue_id, turn_idx = dial_turn.rsplit("_", 1)
                usr_utterance = batch["user_utterance"][i]
                service = batch["service"][i]
                collector[dialogue_id][turn_idx]["utterance"] = usr_utterance
                collector[dialogue_id][turn_idx][service]["predicted_str"] = bs_pred_str
    return dict(collector)
