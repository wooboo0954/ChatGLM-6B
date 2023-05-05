PRE_SEQ_LEN=256
LR=2e-4

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file ptune_data_v1.txt \
    --validation_file AdvertiseGen/dev.json \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/MyGPT-ChatGLM-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

