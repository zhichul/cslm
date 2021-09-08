#!/usr/bin/env bash
set -e
JOB_ID=0
CHEKPOINT=100000
DECODE_MODE=switch_5_count
BEAM_SIZE=200
for i in $(seq 1 4)
do
CUDA_VISIBLE_DEVICES=1 python3 -u /home/blu/jhu/codeswitch/v1.1/src/cslm/main.py \
    --model_name_or_path ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp115-$JOB_ID/checkpoint-${CHEKPOINT}/pytorch_model.bin \
    --encoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp115/encoder-3-12-384-ctx=16.json \
    --decoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp115/decoder-3-12-384-ctx=16.json \
    --softmix_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp115/softmix_config.json \
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
    ${BLU_CORPORA}/.cache/codeswitch/v1.1/syn-g2-exp115 \
    --output_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp115-$JOB_ID \
    --logging_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp115-$JOB_ID-tensorboard \
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
    --decode_output ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp115-$JOB_ID/checkpoint-${CHEKPOINT}/evaluations/${DECODE_MODE}/num-beams-${BEAM_SIZE}/predictions.${i}.json \
    --seed ${i}

done