#!/usr/bin/env bash
JOB_ID=0

CUDA_VISIBLE_DEVICES=0 python3 -u /home/blu/jhu/codeswitch/v1.1/src/cslm/main.py \
    --model_name_or_path ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp72-$JOB_ID/checkpoint-600000/pytorch_model.bin \
    --encoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp72/encoder-3-12-384-ctx=16.json \
    --decoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp72/decoder-3-12-384-ctx=16.json \
    --softmix_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp72/softmix_config.json \
    --train_file \
    ${BLU_CORPORA}/syn/g2/l1-l2/valid-sample.json \
    --validation_file \
    ${BLU_CORPORA}/syn/g2/l1-l2/train-sample.json \
    --l0_tokenizer \
    ${BLU_CORPORA}/syn/g2/l0.0K.json \
    --l1_tokenizer \
    ${BLU_CORPORA}/syn/g2/l1.0K.json \
    --l2_tokenizer \
    ${BLU_CORPORA}/syn/g2/l2.0K.json \
    --cache_dir \
    ${BLU_CORPORA}/.cache/codeswitch/v1.1/syn-g2-exp72 \
    --output_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp72-$JOB_ID \
    --logging_dir \
    ${BLU_ARTIFACTS}/codeswitch/v1.1/mixture-of-softmax-exp72-$JOB_ID-tensorboard \
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
    --eval_mode unigram_precision \
    --eval_output terminal \
    --eval_format human \
    --eval_reduction micro \
    --decode_mode \
    l1_5_l2 \
    --decode_format human \
    --decode_num_beams 10 \
    --decode_do_sample \
    --decode_num_sequences 10 \
    --eval_filter "constrained_decoding_bin_selector(2)" "constrained_decoding_length_selector(5)" \
    --decode_first_n 100 \
#      --decode_output terminal \
    #    --decode_load_cache "eval_debug.json" \


#    --decode_overwrite_output
#        --decode_mode cross_entropy \
#    --decode_output terminal \
#    --decode_format human \
