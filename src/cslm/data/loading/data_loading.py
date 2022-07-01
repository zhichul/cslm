import torch

from cslm.data.loading.preprocessing.tritext import meaning_to_text_preprocessor, asym_meaning_to_text_preprocessor, \
    combined_meaning_to_text_preprocessor
from datasets import load_dataset
import os
import orjson


def readone(file):
    with open(file, "rb") as f:
        line = f.readline()
    return orjson.loads(line)


def load_tritext_dataset(dataset_file=None,
                         preprocessor=None,
                         l0_tokenizer=None,
                         l1_tokenizer=None,
                         l2_tokenizer=None,
                         cache_dir=None,
                         cache_name=None,
                         overwrite_cache=True,
                         num_workers=1):
    """
    Loads a line by line json datasets to integers. Each line of the dataset looks like the following:
    {
        "l0":"verb verb3 obj noun23 mod adj74 subj noun90 mod adj9",
        "l1":"adj9-1 noun90-1 verb3-1 adj74-1 noun23-1",
        "l2":"noun90-2 adj9-2 verb3-2 noun23-2 adj74-2"
    }

    :param dataset_file: line by line json file
    :param preprocessor: logic to turn a list of triples of text into list of training input
                        (integer ids, attention masks, etc)
    :param l0_tokenizer:
    :param l1_tokenizer:
    :param l2_tokenizer:
    :param max_length: maximum length while tokenizing, including [BOS] and [EOS]
    :param cache_dir:
    :param cache_name:
    :param overwrite_cache:
    :param num_workers:
    :return:
    """
    # load json to pyarrow
    dataset = load_dataset("json", data_files={"default": dataset_file}, cache_dir=cache_dir)["default"]

    # function that will map triples of strings to training input format (input_ids, attention_masks, etc)
    if preprocessor == "meaning_to_text":
        prepare = meaning_to_text_preprocessor(l0_tokenizer, l1_tokenizer, l2_tokenizer)
    elif preprocessor == "asym_meaning_to_text":
        prepare = asym_meaning_to_text_preprocessor(l0_tokenizer, l1_tokenizer, l2_tokenizer)
    elif preprocessor == "combined_meaning_to_text":
        prepare = combined_meaning_to_text_preprocessor(l0_tokenizer, l1_tokenizer, l2_tokenizer)
    else:
        raise ValueError(f"Unknown tritext preprocessor: {preprocessor}")

    # if cache name is provided use a cache file
    if cache_dir is not None and cache_name is not None:
        cache_file_name = os.path.join(cache_dir, cache_name)
    else:
        cache_file_name = None

    # dataset is represented as column table indexed by str
    # dataset.map with prepare adds new columns to the table (e.g. input_ids, attention_mask, etc)
    # and we want to remove the initial columns "l0" "l1" and "l2" from the final table
    # that's why we have this list of column names to remove
    rmv_cols = list(readone(dataset_file).keys())
    # preprocess the dataset
    dataset = dataset.map(
        prepare,
        batched=True,
        writer_batch_size=1000,
        batch_size=1000,
        num_proc=num_workers,
        remove_columns=rmv_cols,
        load_from_cache_file=not overwrite_cache,
        keep_in_memory=False,
        cache_file_name=cache_file_name
    )
    return dataset


def default_data_collator(features):
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def encoder_decoder_data_collator_factory(ignore_offset=-1):
    def encoder_decoder_data_collator(features):
        batch = default_data_collator(features)
        if "input_ids_offset" in batch:
            batch["input_ids"][:, 1:] += torch.where(batch["input_ids"][:, 1:] == ignore_offset,0,batch["input_ids_offset"].unsqueeze(-1))
            del batch["input_ids_offset"]
        if "decoder_input_ids_offset" in batch:
            batch["decoder_input_ids"][:, 1:] += torch.where(batch["decoder_input_ids"][:, 1:] == ignore_offset, 0, batch["decoder_input_ids_offset"].unsqueeze(-1))
            batch["labels"][:, 1:] += torch.where(batch["labels"][:, 1:] == ignore_offset, 0, batch["decoder_input_ids_offset"].unsqueeze(-1))
            del batch["decoder_input_ids_offset"]
        batch["labels"] = batch["labels"] * (batch["decoder_attention_mask"]) + (-100) * (1 - batch["decoder_attention_mask"])
        return batch
    return encoder_decoder_data_collator

def no_offset_encoder_decoder_data_collator_factory():
    def encoder_decoder_data_collator(features):
        batch = default_data_collator(features)
        batch["labels"] = batch["labels"] * (batch["decoder_attention_mask"]) + (-100) * (1 - batch["decoder_attention_mask"])
        return batch
    return encoder_decoder_data_collator