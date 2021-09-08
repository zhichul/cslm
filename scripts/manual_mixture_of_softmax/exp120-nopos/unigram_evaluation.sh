#!/usr/bin/env bash
set -e
JOB_ID=0
CHEKPOINT=100000
DECODE_MODE=switch_5_count
BEAM_SIZE=200
for eval_mode in unigram_precision unigram_recall
do
for bin in $(seq 0 5)
do
for i in $(seq 1 4)
do
  CUDA_VISIBLE_DEVICES= python3 -u /home/blu/jhu/codeswitch/v1.1/src/cslm/main.py \
    --device cpu \
    --model_name_or_path ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp120-$JOB_ID/checkpoint-${CHEKPOINT}/pytorch_model.bin \
    --encoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp120/encoder-3-12-384-ctx=16.json \
    --decoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp120/decoder-3-12-384-ctx=16.json \
    --softmix_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp120/softmix_config.json \
    --train_file \
    ${BLU_CORPORA}/syn/g2/l1-l2/valid-sample.json \
    --validation_file \
    ${BLU_CORPORA}/syn/g2/l1-l2/valid-sample.json \
    --l0_tokenizer \
    ${BLU_CORPORA}/syn/g2/l0.0K.json \
    --l1_tokenizer \
    ${BLU_CORPORA}/syn/g2/l1.0K.json \
    --l2_tokenizer \
    ${BLU_CORPORA}/syn/g2/l2.0K.json \
    --cache_dir \
    ${BLU_CORPORA}/.cache/codeswitch/v1.1/syn-g2-exp120 \
    --output_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp120-$JOB_ID \
    --logging_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp120-$JOB_ID-tensorboard \
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
    softmix_lm_head \
    --names \
    lm_head \
    --train_mode \
    mle \
    --max_length \
    16 \
    --dataset_num_workers \
    5 \
    --train_task \
    meaning_to_text \
    --eval_task \
    meaning_to_text \
    --decode_mode \
    ${DECODE_MODE} \
    --decode_format data \
    --decode_num_beams ${BEAM_SIZE} \
    --decode_num_sequences ${BEAM_SIZE} \
    --decode_do_sample \
    --decode_load_cache ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp120-$JOB_ID/checkpoint-${CHEKPOINT}/evaluations/${DECODE_MODE}/num-beams-${BEAM_SIZE}/predictions.${i}.json \
    --eval_mode \
      ${eval_mode} \
    --eval_output \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp120-$JOB_ID/checkpoint-${CHEKPOINT}/evaluations/${DECODE_MODE}/num-beams-${BEAM_SIZE}/evaluations.${i}.bin${bin}.${eval_mode}.json \
    --eval_format data \
    --eval_filter "constrained_decoding_bin_selector(${bin})"
done
shopt -s extglob
ROOT=${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp120-$JOB_ID/checkpoint-${CHEKPOINT}/evaluations/${DECODE_MODE}/num-beams-${BEAM_SIZE}
merge_json ${ROOT}/evaluations.*.bin${bin}.${eval_mode}.json > ${ROOT}/average.bin${bin}.${eval_mode}.json
done
done
