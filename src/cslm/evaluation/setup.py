import os
import sys
from collections import OrderedDict

from cslm.data.loading.tokenizer_loading import combine_wordlevel_tokenizer

from cslm.evaluation.evaluation import EvaluationList, BreakdownEvaluation
from cslm.evaluation.inspections.cross_entropy import CrossEntropyInspection, DualActivationCrossEntropy
from cslm.evaluation.inspections.softmix_cross_attention import SoftmixCrossAttention

from cslm.evaluation.predictions.constrained_decoding import ConstrainedDecoding
from cslm.evaluation.predictions.cross_entropy import CrossEntropyPrediction
from cslm.evaluation.metrics.cross_entropy import CrossEntropyEvaluation
from cslm.evaluation.metrics.length_mismatch import LengthMismatch
from cslm.evaluation.metrics.next_word_pos import SyntheticNextWordPOS
from cslm.evaluation.inspections.softmix_coeff import SoftmixCoeff
from cslm.evaluation.metrics.tok_count import TokCount
from cslm.evaluation.metrics.unigram_evaluation import UnigramLanguageAgnosticRecall, UnigramLanguageAgnosticPrecision
from cslm.evaluation.predictions.gradient_estimation import GradientEstimation
from cslm.evaluation.predictions.sampling import Sampling

from cslm.inference.search_schemes import l1_mixed_l2, l1_3_l2, l1_5_l2, switch_5_percentage, switch_5_count

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
    filters = [eval(filter) for filter in exp_args.decode_filter]
    # setup prediction
    if exp_args.decode_mode.startswith("sample"):
        prediction = Sampling(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
                                         bos_id=bos_id,
                                         eos_ids=eos_ids,
                                         pad_id=pad_id,
                                         vocab_size=vocab_size,
                                         l0_tokenizer=l0_tokenizer,
                                         l1_tokenizer=l1_tokenizer,
                                         l2_tokenizer=l2_tokenizer,
                                         output_file=output_file,
                                         cache_file=cache_file,
                                         filters=filters)
    elif exp_args.decode_mode.startswith("interventional_sample"):
        l1_size = len(l1_tokenizer.get_vocab())
        prediction = Sampling(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
                                         bos_id=bos_id,
                                         eos_ids=eos_ids,
                                         pad_id=pad_id,
                                         vocab_size=vocab_size,
                                         l0_tokenizer=l0_tokenizer,
                                         l1_tokenizer=l1_tokenizer,
                                         l2_tokenizer=l2_tokenizer,
                                         output_file=output_file,
                                         cache_file=cache_file,
                                         filters=filters,
                                         force_langauge=True,
                                         l1_range=slice(4, l1_size, 1),
                                         l2_range=slice(l1_size + 4, vocab_size, 1))
    elif exp_args.decode_mode.startswith("l1_mixed_l2"):
        fn_initial_state = l1_mixed_l2.initial_state_factory()
        fn_update_state = l1_mixed_l2.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab()))
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
                                         cache_file=cache_file,
                                         filters=filters)
    elif exp_args.decode_mode.startswith("l1_3_l2"):
        fn_initial_state = l1_3_l2.initial_state_factory()
        fn_update_state = l1_3_l2.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab()))
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
                                         cache_file=cache_file,
                                         filters=filters)
    elif exp_args.decode_mode.startswith("l1_5_l2"):
        fn_initial_state = l1_5_l2.initial_state_factory()
        fn_update_state = l1_5_l2.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab()))
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
                                         cache_file=cache_file,
                                         filters=filters)
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
            cache_file=cache_file,
            filters=filters)
    elif exp_args.decode_mode == "switch_5_percentage":
        fn_initial_state = switch_5_percentage.initial_state_factory()
        fn_update_state = switch_5_percentage.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab()))
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
                                         cache_file=cache_file,
                                         filters=filters)
    elif exp_args.decode_mode == "switch_5_count":
        fn_initial_state = switch_5_count.initial_state_factory()
        fn_update_state = switch_5_count.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab()))
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
                                         cache_file=cache_file,
                                         filters=filters)
    elif exp_args.decode_mode == "dual_activation_force_language_switch_5_count":
        l1_size = len(l1_tokenizer.get_vocab())
        fn_initial_state = switch_5_count.initial_state_factory()
        fn_update_state = switch_5_count.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab()))
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
                                         cache_file=cache_file,
                                         filters=filters,
                                         dual_activation=True,
                                         dual_activation_force_language=True,
                                         l1_range=slice(4, l1_size, 1),
                                         l2_range=slice(l1_size + 4, vocab_size, 1)
                                         )
    elif exp_args.decode_mode == "dual_activation_switch_5_count":
        fn_initial_state = switch_5_count.initial_state_factory()
        fn_update_state = switch_5_count.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab()))
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
                                         cache_file=cache_file,
                                         filters=filters,
                                         dual_activation=True)
    elif exp_args.decode_mode.startswith("gradient_estimation"):
        l1_size = len(l1_tokenizer.get_vocab())
        prediction = GradientEstimation(model=model,
                                        args=exp_args,
                                        eval_dataset=datasets["validation"],
                                        data_collator=data_collator,
                                        bos_id=bos_id,
                                        eos_ids=eos_ids,
                                        pad_id=pad_id,
                                        vocab_size=vocab_size,
                                        l0_tokenizer=l0_tokenizer,
                                        l1_tokenizer=l1_tokenizer,
                                        l2_tokenizer=l2_tokenizer,
                                        output_file=output_file,
                                        cache_file=cache_file,
                                        l1_range=slice(4, l1_size, 1),
                                        l2_range=slice(l1_size + 4, vocab_size, 1),
                                        filters=filters)
    else:
        raise NotImplementedError
    return prediction


def setup_breakdown_evaluation(prediction=None,
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
        os.makedirs(os.path.dirname(exp_args.eval_output), exist_ok=True)
        output_file = open(exp_args.eval_output, "wb")
    elif exp_args.eval_output != "terminal" and exp_args.eval_format == "human":
        os.makedirs(os.path.dirname(exp_args.eval_output), exist_ok=True)
        output_file = open(exp_args.eval_output, "wt")
    else:
        raise ValueError(f"Unknown output / format: {exp_args.eval_output}/{exp_args.eval_format}")
    filters = [eval(filter) for filter in exp_args.eval_filter]
    factory = lambda: setup_evaluation(prediction=prediction,
                     exp_args=exp_args,
                     l0_tokenizer=l0_tokenizer,
                     l1_tokenizer=l1_tokenizer,
                     l2_tokenizer=l2_tokenizer)
    bkd_evaluation = BreakdownEvaluation(prediction=prediction,
                                     args=exp_args,
                                     output_file=output_file,
                                     filters=filters,
                                     factory=factory)
    for key in exp_args.eval_breakdown:
        bkd_evaluation.add_breakdown(key)
    return bkd_evaluation

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
        os.makedirs(os.path.dirname(exp_args.eval_output), exist_ok=True)
        output_file = open(exp_args.eval_output, "wb")
    elif exp_args.eval_output != "terminal" and exp_args.eval_format == "human":
        os.makedirs(os.path.dirname(exp_args.eval_output), exist_ok=True)
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
    elif exp_args.eval_mode == "next_word_pos":
        evaluation = SyntheticNextWordPOS(prediction=prediction,
                                                      args=exp_args,
                                                      output_file=output_file,
                                                      reduction=exp_args.eval_reduction,
                                                      filters=filters,
                                                      l0_tokenizer=l0_tokenizer,
                                                      l1_tokenizer=l1_tokenizer,
                                                      l2_tokenizer=l2_tokenizer)
    elif exp_args.eval_mode == "tok_count":
        evaluation = TokCount(prediction=prediction,
                              args=exp_args,
                              output_file=output_file,
                              reduction=exp_args.eval_reduction,
                              filters=filters)
    elif exp_args.eval_mode == "length_mismatch":
        evaluation = LengthMismatch(prediction=prediction,
                              args=exp_args,
                              output_file=output_file,
                              reduction=exp_args.eval_reduction,
                              filters=filters)
    else:
        raise NotImplementedError
    return evaluation

def setup_metrics(exp_args=None,
                  model=None,
                  datasets=None,
                  data_collator=None,
                  bos_id=None,
                  eos_ids=None,
                  pad_id=None,
                  vocab_size=None,
                  l0_tokenizer=None,
                  l1_tokenizer=None,
                  l2_tokenizer=None,
                  l1_size=None,
                  l2_size=None):
    validation_metrics = {
        "cross_entropy": {
            "prediction": "cross_entropy",
            "evaluation": "cross_entropy",
            "name": "cross_entropy"
        },
        "interventional_cross_entropy": {
            "prediction": "interventional_cross_entropy",
            "evaluation": "cross_entropy",
            "name": "interventional_cross_entropy"
        },
        "partial_aligned_interventional_cross_entropy": {
            "prediction": "partial_aligned_interventional_cross_entropy",
            "evaluation": "cross_entropy",
            "name": "partial_aligned_interventional_cross_entropy"
        },
        "syn_aware_interventional_cross_entropy": {
            "prediction": "syn_aware_interventional_cross_entropy",
            "evaluation": "cross_entropy",
            "name": "syn_aware_interventional_cross_entropy"
        },
        "unigram_precision": {
            "prediction": "constrained_decoding",
            "evaluation": "unigram_precision",
            "name": "unigram_precision"
        }
    }

    # TODO: move this setup for the shared interventional case to a nicer place
    combined_tokenizer = combine_wordlevel_tokenizer(l1_tokenizer, l2_tokenizer, overlap=True)
    shared_vocab = [(k, v) for k, v in
                    filter(lambda x: not x[0][-1].isnumeric(), combined_tokenizer.get_vocab().items())]
    shared_ids = set(v for k, v in shared_vocab)

    validation_predictions = {
        "cross_entropy": CrossEntropyPrediction(
            model=model,
            args=exp_args,
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            output_file=None,
            bos_id=bos_id,
            eos_ids=eos_ids,
            pad_id=pad_id,
            vocab_size=vocab_size,
            l0_tokenizer=l0_tokenizer,
            l1_tokenizer=l1_tokenizer,
            l2_tokenizer=l2_tokenizer,
            cache_file=None
        ),
        "interventional_cross_entropy": CrossEntropyPrediction(
            model=model,
            args=exp_args,
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            output_file=None,
            bos_id=bos_id,
            eos_ids=eos_ids,
            pad_id=pad_id,
            vocab_size=vocab_size,
            l0_tokenizer=l0_tokenizer,
            l1_tokenizer=l1_tokenizer,
            l2_tokenizer=l2_tokenizer,
            cache_file=None,
            force_langauge=True,
            l1_range=slice(4, l1_size, 1),
            l2_range=slice(l1_size + 4, vocab_size, 1)
        ),
        "partial_aligned_interventional_cross_entropy": CrossEntropyPrediction(
            model=model,
            args=exp_args,
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            output_file=None,
            bos_id=bos_id,
            eos_ids=eos_ids,
            pad_id=pad_id,
            vocab_size=vocab_size,
            l0_tokenizer=l0_tokenizer,
            l1_tokenizer=l1_tokenizer,
            l2_tokenizer=l2_tokenizer,
            cache_file=None,
            force_langauge=True,
            l1_range=[i for i in range(4, l1_size, 1) if i not in shared_ids],
            l2_range=[i for i in range(l1_size + 4, vocab_size, 1) if i not in shared_ids]
        ),
        "syn_aware_interventional_cross_entropy": CrossEntropyPrediction(
            model=model,
            args=exp_args,
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            output_file=None,
            bos_id=bos_id,
            eos_ids=eos_ids,
            pad_id=pad_id,
            vocab_size=vocab_size,
            l0_tokenizer=l0_tokenizer,
            l1_tokenizer=l1_tokenizer,
            l2_tokenizer=l2_tokenizer,
            cache_file=None,
            force_langauge=True,
            l1_range=slice(4, l1_size, 1),
            l2_range=slice(l1_size + 4, vocab_size, 1),
            specialize_decoder_by_language=True,
        ),
        "constrained_decoding": ConstrainedDecoding(model=model,
                                                    args=exp_args,
                                                    eval_dataset=datasets["validation"],
                                                    data_collator=data_collator,
                                                    bos_id=bos_id,
                                                    eos_ids=eos_ids,
                                                    pad_id=pad_id,
                                                    vocab_size=vocab_size,
                                                    fn_initial_state=switch_5_count.initial_state_factory(),
                                                    fn_update_state=switch_5_count.update_state_factory(eos_ids, len(l1_tokenizer.get_vocab())),
                                                    fn_assign_bin=switch_5_count.assign_bin_factory(),
                                                    num_bins=switch_5_count.NUM_BINS,
                                                    do_sample=True,
                                                    l0_tokenizer=l0_tokenizer,
                                                    l1_tokenizer=l1_tokenizer,
                                                    l2_tokenizer=l2_tokenizer,
                                                    output_file=sys.stdout,
                                                    cache_file=None)
    }
    validation_evaluations = {
        "cross_entropy": CrossEntropyEvaluation(prediction=None,
                                                args=exp_args,
                                                output_file=None,
                                                reduction="micro",
                                                filters=list()),
        "unigram_precision": UnigramLanguageAgnosticPrecision(prediction=None,
                                                              args=exp_args,
                                                              output_file=None,
                                                              reduction="micro",
                                                              filters=list(),
                                                              l0_tokenizer=l0_tokenizer,
                                                              l1_tokenizer=l1_tokenizer,
                                                              l2_tokenizer=l2_tokenizer)
    }
    evaluations = OrderedDict()
    for metric in exp_args.metrics:
        metric_meta = validation_metrics[metric]
        pred_key, eval_key, name = metric_meta["prediction"], metric_meta["evaluation"], metric_meta["name"]
        if pred_key not in evaluations:
            evaluations[pred_key] = EvaluationList(prediction=validation_predictions[pred_key],
                                                   args=exp_args,
                                                   output_file=None)
        evaluations[pred_key].add_evaluation(name, validation_evaluations[eval_key])
    return evaluations

def setup_inspection(exp_args=None,
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
    if exp_args.inspect_mode is None:
        return None
    # setup output file
    if exp_args.inspect_output is not None \
            and exp_args.inspect_output != "terminal" \
            and os.path.exists(exp_args.inspect_output) \
            and not exp_args.inspect_overwrite_output:
        raise ValueError(
            f"{exp_args.inspect_output} exists, please set inspect_overwrite_output to true if you wish to overwrite.")
    if exp_args.inspect_output is None:
        output_file = None
    elif exp_args.inspect_output == "terminal" and exp_args.inspect_format == "data":
        output_file = sys.stdout.buffer
    elif exp_args.inspect_output == "terminal" and exp_args.inspect_format == "human":
        output_file = sys.stdout
    elif exp_args.inspect_output != "terminal" and exp_args.inspect_format == "data":
        os.makedirs(os.path.dirname(exp_args.inspect_output), exist_ok=True)
        output_file = open(exp_args.inspect_output, "wb")
    elif exp_args.inspect_output != "terminal" and exp_args.inspect_format == "human":
        os.makedirs(os.path.dirname(exp_args.inspect_output), exist_ok=True)
        output_file = open(exp_args.inspect_output, "wt")
    else:
        raise ValueError(f"Unknown output / format: {exp_args.inspect_output}/{exp_args.inspect_format}")

    if exp_args.inspect_load_cache is not None:
        cache_file = open(exp_args.inspect_load_cache, "rb")
    else:
        cache_file = None

    # setup prediction
    if exp_args.inspect_mode == "softmix_coeff":
        inspection = SoftmixCoeff(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
                                         l0_tokenizer=l0_tokenizer,
                                         l1_tokenizer=l1_tokenizer,
                                         l2_tokenizer=l2_tokenizer,
                                         output_file=output_file,
                                         cache_file=cache_file)
    elif exp_args.inspect_mode == "softmix_cross_attention":
        inspection = SoftmixCrossAttention(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
                                         l0_tokenizer=l0_tokenizer,
                                         l1_tokenizer=l1_tokenizer,
                                         l2_tokenizer=l2_tokenizer,
                                         output_file=output_file,
                                         cache_file=cache_file)
    elif exp_args.inspect_mode == "cross_entropy":
        inspection = CrossEntropyInspection(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
                                         l0_tokenizer=l0_tokenizer,
                                         l1_tokenizer=l1_tokenizer,
                                         l2_tokenizer=l2_tokenizer,
                                         output_file=output_file,
                                         cache_file=cache_file)
    elif exp_args.inspect_mode == "dual_activation_cross_entropy":
        inspection = DualActivationCrossEntropy(model=model,
                                         args=exp_args,
                                         eval_dataset=datasets["validation"],
                                         data_collator=data_collator,
                                         l0_tokenizer=l0_tokenizer,
                                         l1_tokenizer=l1_tokenizer,
                                         l2_tokenizer=l2_tokenizer,
                                         output_file=output_file,
                                         cache_file=cache_file)
    else:
        raise NotImplementedError
    return inspection
