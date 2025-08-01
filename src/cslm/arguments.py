from dataclasses import field, dataclass
from typing import Optional, List


@dataclass
class ExperimentArguments:
    # * * * * * * * * * * * * * * * * * * * * Training and Logging * * * * * * * * * * * * * * * * * * * * * * * * * * #
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    model_name_or_path: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=None, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )

    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )
    train_mode: str = field(default="mle", metadata={
        "help": "training mode, including mle, adversarial_sgda, adversarial_bilevel, etc"
    })
    device: str = field(default="cuda:0")

    # * * * * * * * * * * * * * * * * * * * * Modeling * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    encoder_config: str = field(
        default=None, metadata={"help": "Pretrained encoder config name if creating new model"}
    )
    decoder_config: str = field(
        default=None, metadata={"help": "Pretrained decoder config name if creating new model"}
    )
    softmix_config: str = field(
        default=None, metadata={"help": "Pretrained softmix config name if creating new model"}
    )
    max_length: Optional[int] = field(
        default=64, metadata={"help": "maximum length allowed including special tokens"}
    )
    heads: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "list of heads necessary for training"}
    )
    names: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "list of heads necessary for training"}
    )
    ignore_encoder: Optional[bool] = field(
        default=False, metadata={"help": "ignore the encoder effectively turning it into a language model. (Not exactly, since this just replaces encoder output with zeros)"}
    )
    # * * * * * * * * * * * * * * * * * * * * Data * ** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    l1_tokenizer: str = field(
        default=None, metadata={"help": "Pretrained l1 tokenizer name"}
    )
    l2_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Pretrained l2 tokenizer name"}
    )
    l0_tokenizer: str = field(
        default=None, metadata={"help": "Pretrained l0 tokenizer name"}
    )
    train_file: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "train file(s)"}
    )
    train_weight: Optional[List[float]] = field(
        default_factory=lambda: [1.0], metadata={"help": "Weights of training examples"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Validation file"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache Dir"}
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "overwrite cache"}
    )
    dataset_format: Optional[str] = field(
        default="tritext", metadata={"help": "what format the training / validation data is"}
    )
    train_task: Optional[str] = field(
        default="multilingual", metadata={"help": "what task to train on with a parallel corpus"
                                                  "multilingual: all examples do l1 -> l1, l1 -> l2, l2 -> l1 and l2 -> l2"
                                                  "disjoint_copy: split into half, one half do l1 -> l1, the other do l2 -> l2"
                                                  "disjoint_translate: split into half, one half do l1 -> l2, the other do l2 -> l1"
                                                  "disjoint_copy_and_translate: split into quarter, evenly"                                                  "translate_to_l1"
                                                  "translate_to_l1"
                                                  "translate_to_l2"}
    )
    eval_task: Optional[str] = field(
        default="multilingual", metadata={"help": "what task to train on with a parallel corpus"
                                                  "multilingual: all examples do l1 -> l1, l1 -> l2, l2 -> l1 and l2 -> l2"
                                                  "disjoint_copy: split into half, one half do l1 -> l1, the other do l2 -> l2"
                                                  "disjoint_translate: split into half, one half do l1 -> l2, the other do l2 -> l1"
                                                  "disjoint_copy_and_translate: split into quarter, evenly"
                                                  "translate_to_l1"
                                                  "translate_to_l2"}
    )
    dataset_num_workers: Optional[int] = field(
        default=1, metadata={"help": "number of processes for loading dataset"}
    )
    metrics: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "list of metrics to use for validation during training"}
    )
    monte_carlo_num_sequences: Optional[int] = field(
        default=5, metadata={"help": "Number of beams to return from beam search."}
    )
    # * * * * * * * * * * * * * * * * * * * * Inference  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    decode_mode: Optional[str] = field(
        default=None, metadata={"help": "If set will run decoding, one of constrained_decoding. Otherwise do not run decode."}
    )
    decode_format: Optional[str] = field(
        default="data", metadata={"help": "data, human"}
    )
    decode_output: Optional[str] = field(
        default=None, metadata={"help": "Output of decoding. If set to None means print to terminal."}
    )
    decode_num_beams: Optional[int] = field(
        default=5, metadata={"help": "Beam search parameter."}
    )
    decode_num_sequences: Optional[int] = field(
        default=5, metadata={"help": "Number of beams to return from beam search."}
    )
    decode_display_sequences: Optional[int] = field(
        default=5, metadata={"help": "Number of beams to return from beam search."}
    )
    decode_do_sample: Optional[bool] = field(
        default=False, metadata={"help": "Whether to do stochastic decoding."}
    )
    decode_load_cache: Optional[str] = field(
        default=None, metadata={"help": "Load from cache instead of actually decoding"}
    )
    decode_first_n: Optional[int] = field(
        default=None, metadata={"help": "for debugging, only predict the first n examples"}
    )
    decode_overwrite_output: bool = field(
        default=False, metadata={"help": "Overwrite the content of the decode output file."},
    )
    decode_repetitions: Optional[int] = field(
        default=1, metadata={"help": "how many times to run each prediction"}
    )
    # * * * * * * * * * * * * * * * * * * * * Evaluation Inference  * * * * * * * * * * * * * * * * * * * * * * * * * #
    eval_mode: Optional[str] = field(
        default=None, metadata={"help": "If set will run evaluation, one of cross_entropy. Otherwise do not run decode."}
    )
    eval_output: Optional[str] = field(
        default=None, metadata={"help": "Eval output file"}
    )
    eval_format: Optional[str] = field(
        default="data", metadata={"help": "one of human/data."}
    )
    eval_reduction: Optional[str] = field(
        default="macro", metadata={"help": "one of {'macro', 'micro', 'none'}"}
    )
    # * * * * * * * * * * * * * * * * * * * * Custom Evaluation  * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    eval_filter: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "filter prediction outputs"}
    )
    eval_breakdown: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "breakdown prediction outputs"}
    )
    decode_filter: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "filter prediction outputs"}
    )
    # * * * * * * * * * * * * * * * * * * * * Inspection   * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    inspect_mode: Optional[str] = field(
        default=None, metadata={"help": "If set will run inspection. Otherwise do not run decode."}
    )
    inspect_output: Optional[str] = field(
        default=None, metadata={"help": "inspection output file"}
    )
    inspect_format: Optional[str] = field(
        default="data", metadata={"help": "one of human/data."}
    )
    inspect_load_cache: Optional[str] = field(
        default=None, metadata={"help": "Load from cache instead of actually inspecting"}
    )
    inspect_live: Optional[bool] = field(
        default=False, metadata={"help": "Live interactive inspection"}
    )
    inspect_console: Optional[bool] = field(
        default=False, metadata={"help": "Live console inspection"}
    )
    inspect_expose: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "Exposure patterns"}
    )
    # * * * * * * * * * * * * * * * * * * * * Flags   * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    switch_position_1_2: Optional[bool] = field(
        default=False
    )