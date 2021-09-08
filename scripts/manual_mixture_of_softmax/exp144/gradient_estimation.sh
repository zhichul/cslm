#!/usr/bin/env bash
JOB_ID=0
CHEKPOINT=600000
DECODE_MODE=switch_5_count
BEAM_SIZE=200
for i in 1 2 4 8 16 32 64 128 256 512
do
CUDA_VISIBLE_DEVICES=1 python3 -u /home/blu/jhu/codeswitch/v1.1/src/cslm/main.py \
    --model_name_or_path \
    /export/a01/artifacts/codeswitch/v1.1/mixture-of-softmax-exp144-0/checkpoint-1000/pytorch_model.bin \
    --encoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp144/encoder-3-12-384-ctx=16.json \
    --decoder_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp144/decoder-3-12-384-ctx=16.json \
    --softmix_config \
    /home/blu/jhu/codeswitch/v1.1/scripts/manual_mixture_of_softmax/configs/exp144/softmix_config.json \
    --train_file \
    /export/a01/corpora/syn/g2/l1-l2/valid-sample.json \
    --validation_file \
    /export/a01/corpora/syn/g2/l1-l2/valid-sample.json \
    --l0_tokenizer \
    /export/a01/corpora/syn/g2/l0.0K.json \
    --l1_tokenizer \
    /export/a01/corpora/syn/g2/l1.0K.json \
    --l2_tokenizer \
    /export/a01/corpora/syn/g2/l2.0K.json \
    --cache_dir \
    /export/a01/corpora/.cache/codeswitch/v1.1/syn-g2-exp144 \
    --output_dir \
    /export/a01/artifacts/codeswitch/v1.1/mixture-of-softmax-exp144-0 \
    --logging_dir \
    /export/a01/artifacts/codeswitch/v1.1/mixture-of-softmax-exp144-0-tensorboard \
    --overwrite_cache \
    --overwrite_output_dir \
    --max_steps \
    100000 \
    --learning_rate \
    5e-5 \
    --per_device_train_batch_size \
    16 \
    --gradient_accumulation_steps \
    1 \
    --per_device_eval_batch_size \
    1 \
    --eval_accumulation_steps \
    1 \
    --logging_steps \
    100 \
    --eval_steps \
    500 \
    --save_steps \
    500 \
    --heads \
    softmix_lm_head \
    --names \
    lm_head \
    --train_mode \
    ebm_interventional_mle \
    --max_length \
    16 \
    --train_task \
    meaning_to_text \
    --eval_task \
    meaning_to_text \
    --decode_mode \
    gradient_estimation \
    --decode_format \
    data \
    --decode_output \
    /export/a01/artifacts/codeswitch/v1.1/mixture-of-softmax-exp144-0/checkpoint-1000/gradient_estimation.samples=${i} \
    --decode_first_n \
    10 \
    --decode_repetitions \
    10 \
    --monte_carlo_num_sequences \
    ${i} \
    --per_device_eval_batch_size \
    1

done