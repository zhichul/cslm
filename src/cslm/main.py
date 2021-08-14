import os
import sys

import torch
from datasets import DatasetDict
from torch.utils.data import ConcatDataset
from transformers import HfArgumentParser, AdamW
from transformers.utils import logging

from cslm.arguments import ExperimentArguments
from cslm.data.loading.tokenizer_loading import load_and_setup_tokenizer
from cslm.evaluation.setup import setup_prediction, setup_evaluation, setup_metrics, setup_inspection
from cslm.modeling.configuration import Config, EncoderDecoderConfig
from cslm.modeling.encoder_decoder import EncoderDecoder
from cslm.modeling.head import HeadBuilder
from cslm.modeling.softmix import SoftmixOutputLayer
from cslm.modeling.transformer import TransformerLMHead
from cslm.training.mle_trainer import MLETrainer
from cslm.training.utils import get_linear_schedule_with_warmup
from cslm.utils import set_seed, seq_numel
from cslm.data.loading.data_loading import load_tritext_dataset, encoder_decoder_data_collator_factory
from grid_utils import acquire_all_available_gpu


def main():
    # * * * * * * * * * * * * * * * * * * * * CMD SETUP START * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    acquire_all_available_gpu()
    logger = logging.get_logger(__name__)

    parser = HfArgumentParser(ExperimentArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        exp_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        exp_args, = parser.parse_args_into_dataclasses()

    set_seed(exp_args.seed)

    logger.info(exp_args)
    # * * * * * * * * * * * * * * * * * * * * CMD SETUP END * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * TOKENIZER SETUP START * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    l0_tokenizer = load_and_setup_tokenizer(exp_args.l0_tokenizer, max_length=exp_args.max_length, pad_token="[PAD]")
    l1_tokenizer = load_and_setup_tokenizer(exp_args.l1_tokenizer, max_length=exp_args.max_length, pad_token="[PAD]")
    l2_tokenizer = load_and_setup_tokenizer(exp_args.l2_tokenizer, max_length=exp_args.max_length, pad_token="[PAD]")

    # some useful globals
    bos_id = l1_tokenizer.token_to_id("[BOS]")
    eos_ids = [l1_tokenizer.token_to_id("[EOS]")]
    pad_id = l1_tokenizer.token_to_id("[PAD]")
    vocab_size = len(l1_tokenizer.get_vocab()) + len(l2_tokenizer.get_vocab())
    l1_size = len(l1_tokenizer.get_vocab())
    l2_size = len(l2_tokenizer.get_vocab())
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
    else:
        raise NotImplementedError

    # collator
    data_collator = encoder_decoder_data_collator_factory(ignore_offset=l2_tokenizer.token_to_id("[EOS]"))
    # * * * * * * * * * * * * * * * * * * * * DATASET SETUP END * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * MODEL SETUP START * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    encoder_config = Config.from_json(exp_args.encoder_config)
    decoder_config = Config.from_json(exp_args.decoder_config)
    encoder_decoder_config = EncoderDecoderConfig(encoder_config=encoder_config, decoder_config=decoder_config)
    softmix_config = Config.from_json(exp_args.softmix_config) if exp_args.softmix_config else None

    if exp_args.train_task in ["meaning_to_text", "asym_meaning_to_text"]:
        src_vocab_size = l0_tokenizer.get_vocab_size()
        target_vocab_size = l1_tokenizer.get_vocab_size() + l2_tokenizer.get_vocab_size()
        encoder_config.vocab_size = src_vocab_size
        decoder_config.vocab_size = target_vocab_size
        if softmix_config is not None:
            softmix_config.vocab_size = target_vocab_size
            softmix_config.l1_vocab_size = l1_size
            softmix_config.l2_vocab_size = l2_size
        base_model = EncoderDecoder(encoder_decoder_config)
    else:
        raise NotImplementedError

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
        d = torch.load(exp_args.model_name_or_path, map_location=torch.device("cuda") if torch.cuda.is_available()
                                                            else torch.device("cpu"))
        model.load_state_dict(d, strict=True)
        logger.info(f"loaded from {exp_args.model_name_or_path}")

        if exp_args.switch_position_1_2:
            logger.info(f"switching wpe 1 and 2")
            x1 = model.base_model.decoder.word_position_embed.weight[1].clone().detach()
            x2 = model.base_model.decoder.word_position_embed.weight[2].clone().detach()
            model.base_model.decoder.word_position_embed.weight[2] = x1
            model.base_model.decoder.word_position_embed.weight[1] = x2

    # setup validation evaluation
    evaluations = setup_metrics(exp_args=exp_args,
                                  model=model,
                                  datasets=datasets,
                                  data_collator=data_collator,
                                  bos_id=bos_id,
                                  eos_ids=eos_ids,
                                  pad_id=pad_id,
                                  vocab_size=vocab_size,
                                  l0_tokenizer=l0_tokenizer,
                                  l1_tokenizer=l1_tokenizer,
                                  l2_tokenizer=l2_tokenizer,
                                  l1_size=l1_size,
                                  l2_size=l2_size)

    if exp_args.train_mode == "mle":
        trainer = MLETrainer(
            model=model,
            args=exp_args,
            train_dataset=datasets["train"],
            data_collator=data_collator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataset_weights=exp_args.train_weight,
            train_dataset_lengths=train_dataset_lengths,
            evaluations=evaluations
        )
    elif exp_args.train_mode == "interventional_mle":
        trainer = MLETrainer(
            model=model,
            args=exp_args,
            train_dataset=datasets["train"],
            data_collator=data_collator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataset_weights=exp_args.train_weight,
            train_dataset_lengths=train_dataset_lengths,
            evaluations=evaluations,
            force_langauge=True,
            vocab_size=vocab_size,
            l1_range=slice(4, l1_size, 1),
            l2_range=slice(l1_size + 4, vocab_size, 1)
        )
    else:
        raise NotImplementedError
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
    prediction = setup_prediction(exp_args=exp_args,
                     model=model,
                     datasets=datasets,
                     data_collator=data_collator,
                     l0_tokenizer=l0_tokenizer,
                     l1_tokenizer=l1_tokenizer,
                     l2_tokenizer=l2_tokenizer,
                     vocab_size=vocab_size,
                     bos_id=bos_id,
                     eos_ids=eos_ids,
                     pad_id=pad_id)

    evaluation = setup_evaluation(prediction=prediction,
                     exp_args=exp_args,
                     l0_tokenizer=l0_tokenizer,
                     l1_tokenizer=l1_tokenizer,
                     l2_tokenizer=l2_tokenizer)
    if exp_args.decode_mode is not None and exp_args.eval_mode is None:
        # decode only mode
        logger.info("Running prediction only mode.")
        prediction.predict_and_log()
    elif exp_args.decode_mode is not None and exp_args.eval_mode is not None:
        # decode and eval mode
        logger.info("Running prediction and evaluation.")
        evaluation.evaluate_and_log()
    # * * * * * * * * * * * * * * * * * * * * INFERENCE END * * * * * * * * ** * * * * * * * * * * * * * * * * * * * #
    #
    #
    #
    # * * * * * * * * * * * * * * * * * * * * INSPECTION START * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    if exp_args.inspect_mode is not None:
        inspection = setup_inspection(exp_args=exp_args,
                         model=model,
                         datasets=datasets,
                         data_collator=data_collator,
                         l0_tokenizer=l0_tokenizer,
                         l1_tokenizer=l1_tokenizer,
                         l2_tokenizer=l2_tokenizer)
        inspection.inspect_and_log()
if __name__ == "__main__":
    main()