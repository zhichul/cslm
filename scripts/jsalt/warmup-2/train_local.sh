#!/usr/bin/env bash
JOB_ID=0

NAME=warmup-2

CUDA_VISIBLE_DEVICES=0 python3 -u /home/blu/jhu/codeswitch/v1.1/src/cslm/main.py \
    --encoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/jsalt/${NAME}/encoder.json \
    --decoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/jsalt/${NAME}/decoder.json \
    --train_file \
    ${BLU_CORPORA}/seame/seame_post1/data/raw.json \
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
    --do_train \
    --do_eval \
    --max_steps \
    30000 \
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
    combined_bitext_to_text