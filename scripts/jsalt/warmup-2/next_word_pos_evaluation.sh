#!/usr/bin/env bash
set -e
JOB_ID=0
NAME=warmup-2
CHEKPOINT=300000
DECODE_MODE=switch_5_count
BEAM_SIZE=200
for eval_mode in next_word_pos
do
for bin in $(seq 0 5)
do
for i in $(seq 1 4)
do
CUDA_VISIBLE_DEVICES= python3 -u /home/blu/jhu/codeswitch/v1.1/src/cslm/main.py \
    --device cpu \
    --model_name_or_path ${BLU_ARTIFACTS}/codeswitch/v1.1/jsalt-${NAME}-$JOB_ID/checkpoint-${CHEKPOINT}/pytorch_model.bin \
    --encoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/jsalt/${NAME}/encoder.json \
    --decoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/jsalt/${NAME}/decoder.json \
    --train_file \
    ${BLU_CORPORA}/seame/seame_post1/data/sample.json \
    --validation_file \
    ${BLU_CORPORA}/seame/seame_post1/data/sample.json \
    --l0_tokenizer \
    ${BLU_CORPORA}/seame/seame_post1/combined.6K.json \
    --l1_tokenizer \
    ${BLU_CORPORA}/seame/seame_post1/combined.6K.json \
    --l2_tokenizer \
    ${BLU_CORPORA}/seame/seame_post1/combined.6K.json \
    --cache_dir \
    ${BLU_CORPORA}/.cache/codeswitch/v1.1/jsalt-${NAME} \
    --output_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/jsalt-${NAME}-$JOB_ID \
    --logging_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/jsalt-${NAME}-$JOB_ID-tensorboard \
    --overwrite_cache \
    --overwrite_output_dir \
    --max_steps \
    300000 \
    --learning_rate \
    5e-5 \
    --per_device_train_batch_size \
    64 \
    --gradient_accumulation_steps \
    1 \
    --per_device_eval_batch_size \
    1 \
    --eval_accumulation_steps \
    1 \
    --logging_steps \
    1000 \
    --eval_steps \
    2000 \
    --save_steps \
    5000 \
    --heads \
    softmax_lm_head \
    --names \
    lm_head \
    --train_mode \
    mle \
    --max_length \
    32 \
    --dataset_num_workers \
    5 \
    --train_task \
    combined_bitext_to_text \
    --eval_task \
    combined_bitext_to_text \
    --decode_mode \
    ${DECODE_MODE} \
    --decode_format data \
    --decode_num_beams ${BEAM_SIZE} \
    --decode_num_sequences ${BEAM_SIZE} \
    --decode_do_sample \
    --decode_load_cache ${BLU_ARTIFACTS}/codeswitch/v1.1/jsalt-${NAME}-$JOB_ID/checkpoint-${CHEKPOINT}/evaluations/${DECODE_MODE}/num-beams-${BEAM_SIZE}/predictions.${i}.json \
    --eval_mode \
      ${eval_mode} \
    --eval_output \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/jsalt-${NAME}-$JOB_ID/checkpoint-${CHEKPOINT}/evaluations/${DECODE_MODE}/num-beams-${BEAM_SIZE}/evaluations.${i}.bin${bin}.${eval_mode}.json \
    --eval_format data \
    --eval_filter "constrained_decoding_bin_selector(${bin})" \
    --eval_reduction micro
done
shopt -s extglob
ROOT=${BLU_ARTIFACTS}/codeswitch/v1.1/jsalt-${NAME}-$JOB_ID/checkpoint-${CHEKPOINT}/evaluations/${DECODE_MODE}/num-beams-${BEAM_SIZE}
merge_json ${ROOT}/evaluations.*.bin${bin}.${eval_mode}.json > ${ROOT}/average.bin${bin}.${eval_mode}.json
done
done
