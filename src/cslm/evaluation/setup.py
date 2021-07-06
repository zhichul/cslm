import os
import sys

from cslm.evaluation.constrained_decoding import ConstrainedDecoding
from cslm.evaluation.cross_entropy import CrossEntropyPrediction, CrossEntropyEvaluation
from cslm.evaluation.unigram_evaluation import UnigramLanguageAgnosticRecall, UnigramLanguageAgnosticPrecision

from cslm.inference.search_schemes import l1_mixed_l2, l1_3_l2, l1_5_l2, switch_5_percentage, switch_5_count
from cslm.evaluation.constrained_decoding import constrained_decoding_bin_selector, constrained_decoding_length_selector


def setup_prediction(exp_args=None,
                     model=None,
                     datasets=None,
                     data_collator=None,
                     l0_tokenizer=None,
                     l1_tokenizer=None,
                     l2_tokenizer=None,
                     vocab_size=None,
                     bos_id=None,
                     eos_ids=None,
                     pad_id=None):
    if exp_args.decode_mode is None:
        return None
    # setup output file
    if exp_args.decode_output is not None \
            and exp_args.decode_output != "terminal" \
            and os.path.exists(exp_args.decode_output) \
            and not exp_args.decode_overwrite_output:
        raise ValueError(
            f"{exp_args.decode_output} exists, please set decode_overwrite_output to true if you wish to overwrite.")
    if exp_args.decode_output is None:
        output_file = None
    elif exp_args.decode_output == "terminal" and exp_args.decode_format == "data":
        output_file = sys.stdout.buffer
    elif exp_args.decode_output == "terminal" and exp_args.decode_format == "human":
        output_file = sys.stdout
    elif exp_args.decode_output != "terminal" and exp_args.decode_format == "data":
        os.makedirs(os.path.dirname(exp_args.decode_output), exist_ok=True)
        output_file = open(exp_args.decode_output, "wb")
    elif exp_args.decode_output != "terminal" and exp_args.decode_format == "human":
        os.makedirs(os.path.dirname(exp_args.decode_output), exist_ok=True)
        output_file = open(exp_args.decode_output, "wt")
    else:
        raise ValueError(f"Unknown output / format: {exp_args.decode_output}/{exp_args.decode_format}")

    if exp_args.decode_load_cache is not None:
        cache_file = open(exp_args.decode_load_cache, "rb")
    else:
        cache_file = None

    # setup prediction
    if exp_args.decode_mode.startswith("l1_mixed_l2"):
        fn_initial_state = l1_mixed_l2.initial_state_factory()
        fn_update_state = l1_mixed_l2.update_state_factory(eos_ids)
        fn_assign_bin = l1_mixed_l2.assign_bin_factory()
        num_bins = l1_mixed_l2.NUM_BINS
        do_sample = exp_args.decode_do_sample
        prediction = ConstrainedDecoding(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
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
                                         output_file=output_file,
                                         cache_file=cache_file)
    elif exp_args.decode_mode.startswith("l1_3_l2"):
        fn_initial_state = l1_3_l2.initial_state_factory()
        fn_update_state = l1_3_l2.update_state_factory(eos_ids)
        fn_assign_bin = l1_3_l2.assign_bin_factory()
        num_bins = l1_3_l2.NUM_BINS
        do_sample = exp_args.decode_do_sample
        prediction = ConstrainedDecoding(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
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
                                         output_file=output_file,
                                         cache_file=cache_file)
    elif exp_args.decode_mode.startswith("l1_5_l2"):
        fn_initial_state = l1_5_l2.initial_state_factory()
        fn_update_state = l1_5_l2.update_state_factory(eos_ids)
        fn_assign_bin = l1_5_l2.assign_bin_factory()
        num_bins = l1_5_l2.NUM_BINS
        do_sample = exp_args.decode_do_sample
        prediction = ConstrainedDecoding(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
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
                                         output_file=output_file,
                                         cache_file=cache_file)
    elif exp_args.decode_mode == "cross_entropy":
        prediction = CrossEntropyPrediction(
            model=model,
            args=exp_args,
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            output_file=output_file,
            bos_id=bos_id,
            eos_ids=eos_ids,
            pad_id=pad_id,
            vocab_size=vocab_size,
            l0_tokenizer=l0_tokenizer,
            l1_tokenizer=l1_tokenizer,
            l2_tokenizer=l2_tokenizer,
            cache_file=cache_file
        )
    elif exp_args.decode_mode == "switch_5_percentage":
        fn_initial_state = switch_5_percentage.initial_state_factory()
        fn_update_state = switch_5_percentage.update_state_factory(eos_ids)
        fn_assign_bin = switch_5_percentage.assign_bin_factory()
        num_bins = switch_5_percentage.NUM_BINS
        do_sample = exp_args.decode_do_sample
        prediction = ConstrainedDecoding(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
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
                                         output_file=output_file,
                                         cache_file=cache_file)
    elif exp_args.decode_mode == "switch_5_count":
        fn_initial_state = switch_5_count.initial_state_factory()
        fn_update_state = switch_5_count.update_state_factory(eos_ids)
        fn_assign_bin = switch_5_count.assign_bin_factory()
        num_bins = switch_5_count.NUM_BINS
        do_sample = exp_args.decode_do_sample
        prediction = ConstrainedDecoding(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
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
                                         output_file=output_file,
                                         cache_file=cache_file)
    else:
        raise NotImplementedError
    return prediction


def setup_evaluation(prediction=None,
                     exp_args=None,
                     l0_tokenizer=None,
                     l1_tokenizer=None,
                     l2_tokenizer=None):
    if exp_args.eval_mode is None:
        return None
    if exp_args.eval_output is None:
        output_file = None
    elif exp_args.eval_output == "terminal" and exp_args.eval_format == "data":
        output_file = sys.stdout.buffer
    elif exp_args.eval_output == "terminal" and exp_args.eval_format == "human":
        output_file = sys.stdout
    elif exp_args.eval_output != "terminal" and exp_args.eval_format == "data":
        os.makedirs(os.path.dirname(exp_args.decode_output), exist_ok=True)
        output_file = open(exp_args.eval_output, "wb")
    elif exp_args.eval_output != "terminal" and exp_args.eval_format == "human":
        os.makedirs(os.path.dirname(exp_args.decode_output), exist_ok=True)
        output_file = open(exp_args.eval_output, "wt")
    else:
        raise ValueError(f"Unknown output / format: {exp_args.eval_output}/{exp_args.eval_format}")
    filters = [eval(filter) for filter in exp_args.eval_filter]
    if exp_args.eval_mode == "cross_entropy":
        evaluation = CrossEntropyEvaluation(prediction=prediction,
                                            args=exp_args,
                                            output_file=output_file,
                                            reduction=exp_args.eval_reduction,
                                            filters=filters)
    elif exp_args.eval_mode == "unigram_precision":
        evaluation = UnigramLanguageAgnosticPrecision(prediction=prediction,
                                                      args=exp_args,
                                                      output_file=output_file,
                                                      reduction=exp_args.eval_reduction,
                                                      filters=filters,
                                                      l0_tokenizer=l0_tokenizer,
                                                      l1_tokenizer=l1_tokenizer,
                                                      l2_tokenizer=l2_tokenizer)
    elif exp_args.eval_mode == "unigram_recall":
        evaluation = UnigramLanguageAgnosticRecall(prediction=prediction,
                                                      args=exp_args,
                                                      output_file=output_file,
                                                      reduction=exp_args.eval_reduction,
                                                      filters=filters,
                                                      l0_tokenizer=l0_tokenizer,
                                                      l1_tokenizer=l1_tokenizer,
                                                      l2_tokenizer=l2_tokenizer)
    else:
        raise NotImplementedError
    return evaluation
