PRE_SEQ_LEN=256

CUDA_VISIBLE_DEVICES=0 python3 /content/ChatGLM-6B/ptuning/web_demo.py \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint /content/drive/MyDrive/models/MyGPT-ChatGLM-256-2e-2 \
    --pre_seq_len $PRE_SEQ_LEN
