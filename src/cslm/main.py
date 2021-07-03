import os
import sys

import orjson
import torch
from datasets import DatasetDict
from torch import nn
from torch.utils.data import ConcatDataset
from transformers import HfArgumentParser, AdamW
from transformers.utils import logging

from cslm.arguments import ExperimentArguments
from cslm.data.loading.tokenizer_loading import load_and_setup_tokenizer
from cslm.evaluation.constrained_decoding import ConstrainedDecoding
from cslm.modeling.configuration import Config, EncoderDecoderConfig
from cslm.modeling.encoder_decoder import EncoderDecoder
from cslm.modeling.head import HeadBuilder
from cslm.modeling.softmix import SoftmixOutputLayer
from cslm.modeling.transformer import TransformerLMHead
from cslm.training.mle_trainer import MLETrainer
from cslm.training.utils import get_linear_schedule_with_warmup
from cslm.utils import set_seed, seq_numel, decode_input, decode_output
from cslm.data.loading.data_loading import load_tritext_dataset, encoder_decoder_data_collator_factory
from grid_utils import acquire_all_available_gpu

import cslm.inference.search_schemes.l1_mixed_l2 as l1_mixed_l2

def main():
    # * * * * * * * * * * * * * * * * * * * * CMD SETUP START * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    acquire_all_available_gpu()
    logger = logging.get_logger(__name__)

    set_seed(42)
    parser = HfArgumentParser(ExperimentArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        exp_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        exp_args, = parser.parse_args_into_dataclasses()
    # * * * * * * * * * * * * * * * * * * * * CMD SETUP END * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * TOKENIZER SETUP START * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    l0_tokenizer = load_and_setup_tokenizer(exp_args.l0_tokenizer, max_length=exp_args.max_length, pad_token="[PAD]")
    l1_tokenizer = load_and_setup_tokenizer(exp_args.l1_tokenizer, max_length=exp_args.max_length, pad_token="[PAD]")
    l2_tokenizer = load_and_setup_tokenizer(exp_args.l2_tokenizer, max_length=exp_args.max_length, pad_token="[PAD]")
    # * * * * * * * * * * * * * * * * * * * * TOKENIZER SETUP END * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * DATASET SETUP START * * * * * * * * * * * * * * * * * * * * * * * * * * *  * #
    if exp_args.dataset_format == "tritext":
        valid_dataset = load_tritext_dataset(dataset_file=exp_args.validation_file,
                                             preprocessor=exp_args.eval_task,
                                             l0_tokenizer=l0_tokenizer,
                                             l1_tokenizer=l1_tokenizer,
                                             l2_tokenizer=l2_tokenizer,
                                             cache_dir=os.path.join(exp_args.cache_dir, "valid"),
                                             overwrite_cache=exp_args.overwrite_cache,
                                             num_workers=exp_args.dataset_num_workers)
        train_datasets = []
        train_dataset_lengths = []

        for train_i, train_file in enumerate(exp_args.train_file):
            train_sub_dataset = load_tritext_dataset(dataset_file=train_file,
                                                     preprocessor=exp_args.train_task,
                                                     l0_tokenizer=l0_tokenizer,
                                                     l1_tokenizer=l1_tokenizer,
                                                     l2_tokenizer=l2_tokenizer,
                                                     cache_dir=os.path.join(exp_args.cache_dir, f"dataset-{train_i}"),
                                                     overwrite_cache=exp_args.overwrite_cache,
                                                     num_workers=exp_args.dataset_num_workers)
            train_datasets.append(train_sub_dataset)
            train_dataset_lengths.append(len(train_sub_dataset))
            logger.info(f"Loaded {train_file} with {len(train_sub_dataset)} examples, and going to sample with weight {exp_args.train_weight[train_i]}.")
        train_dataset = ConcatDataset(train_datasets)
        datasets = DatasetDict({"train":train_dataset, "validation": valid_dataset})
    # * * * * * * * * * * * * * * * * * * * * DATASET SETUP END * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * MODEL SETUP START * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    encoder_config = Config.from_json(exp_args.encoder_config)
    decoder_config = Config.from_json(exp_args.decoder_config)
    encoder_decoder_config = EncoderDecoderConfig(encoder_config=encoder_config, decoder_config=decoder_config)
    softmix_config = Config.from_json(exp_args.softmix_config) if exp_args.softmix_config else None

    if exp_args.train_task == "meaning_to_text":
        src_vocab_size = l0_tokenizer.get_vocab_size()
        target_vocab_size = l1_tokenizer.get_vocab_size() + l2_tokenizer.get_vocab_size()
        encoder_config.vocab_size = src_vocab_size
        decoder_config.vocab_size = target_vocab_size
        if softmix_config is not None:
            softmix_config.vocab_size = target_vocab_size
        base_model = EncoderDecoder(encoder_decoder_config)

    logger.info(f"Encoder Size: {seq_numel(tuple(base_model.encoder.parameters()))}")
    logger.info(f"Decoder Size: {seq_numel(tuple(base_model.decoder.parameters()))}")
    logger.info(f"Embedding Size (per 1 embedding module): {base_model.encoder.get_input_embeddings().num_embeddings * base_model.encoder.get_input_embeddings().embedding_dim}")
    heads = {
        "softmax_lm_head": TransformerLMHead(decoder_config),
        "softmix_lm_head": SoftmixOutputLayer(softmix_config) if softmix_config else None,
    }

    builder = HeadBuilder()
    builder.set_base_model(base_model)
    assert len(exp_args.heads) == len(exp_args.names)
    for head, name in zip(exp_args.heads, exp_args.names):
        builder.add_head(heads[head], name)
        logger.info(f"{name} Size: {seq_numel(tuple(heads[head].parameters()))}")
    model = builder.build()
    # * * * * * * * * * * * * * * * * * * * * MODEL SETUP END * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    #
    #
    ## * * * * * * * * * * * * * * * * * * * * OPTIMIZER SETUP START * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": exp_args.weight_decay,
            "lr": exp_args.learning_rate
        },

        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": exp_args.learning_rate
        },
    ]
    optimizer_cls = AdamW
    optimizer_kwargs = {
        "betas": (exp_args.adam_beta1, exp_args.adam_beta2),
        "eps": exp_args.adam_epsilon,
    }

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, exp_args.max_steps)

    # * * * * * * * * * * * * * * * * * * * * OPTIMIZER SETUP END * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * TRAINING SETUP START * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    if exp_args.model_name_or_path:
        d = torch.load(exp_args.model_name_or_path)
        model.load_state_dict(d, strict=True)
        logger.info(f"loaded from {exp_args.model_name_or_path}")
    if exp_args.train_mode == "mle":
        trainer = MLETrainer(
            model=model,
            args=exp_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=encoder_decoder_data_collator_factory(ignore_offset=l2_tokenizer.token_to_id("[EOS]")),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataset_weights=exp_args.train_weight,
            train_dataset_lengths=train_dataset_lengths,
        )
    # * * * * * * * * * * * * * * * * * * * * TRAINING SETUP END * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * TRAINING START * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    if exp_args.do_train:
        trainer.train()
    # * * * * * * * * * * * * * * * * * * * * TRAINING END * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * INFERENCE START * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    if exp_args.decode_mode is not None:
        # setup output file
        if exp_args.decode_output is None and exp_args.decode_format == "data":
            output_file = sys.stdout.buffer
        elif exp_args.decode_output is None and exp_args.decode_format == "human":
            output_file = sys.stdout
        elif exp_args.decode_output is not None and exp_args.decode_format == "data":
            output_file = open(exp_args.decode_output, "wb")
        elif exp_args.decode_output is not None and exp_args.decode_format == "human":
            output_file = open(exp_args.decode_output, "wt")

        # setup evaluation
        if exp_args.decode_mode.startswith("l1_mixed_l2"):
            bos_id = l1_tokenizer.token_to_id("[BOS]")
            eos_ids = [l1_tokenizer.token_to_id("[EOS]")]
            pad_id = l1_tokenizer.token_to_id("[PAD]")
            vocab_size = len(l1_tokenizer.get_vocab()) + len(l2_tokenizer.get_vocab())
            fn_initial_state = l1_mixed_l2.initial_state_factory()
            fn_update_state = l1_mixed_l2.update_state_factory(eos_ids)
            fn_assign_bin = l1_mixed_l2.assign_bin_factory()
            num_bins = l1_mixed_l2.NUM_BINS
            do_sample = exp_args.decode_do_sample
            evaluation = ConstrainedDecoding(model=model,
                                             args=exp_args,
                                             eval_dataset=datasets["validation"],
                                             data_collator=encoder_decoder_data_collator_factory(
                                                                    ignore_offset=l2_tokenizer.token_to_id("[EOS]")),
                                             bos_id=bos_id,
                                             eos_ids=eos_ids,
                                             pad_id=pad_id,
                                             vocab_size=vocab_size,
                                             fn_initial_state=fn_initial_state,
                                             fn_update_state=fn_update_state,
                                             fn_assign_bin=fn_assign_bin,
                                             num_bins=num_bins,
                                             do_sample=do_sample,
                                             l0_tokenizer=l0_tokenizer,
                                             l1_tokenizer=l1_tokenizer,
                                             l2_tokenizer=l2_tokenizer,
                                             output_file=output_file)
            evaluation.predict_and_log()

if __name__ == "__main__":
    main()