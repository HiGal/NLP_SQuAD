#sketchy module
python ./examples/run_cls.py \
    --model_name_or_path albert-base-v2 \
    --task_name squad \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir DATA_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size=6   \
    --per_gpu_eval_batch_size=8   \
    --warmup_steps=814 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --output_dir squad/cls_squad2_albert-base-v2\
    --save_steps 2500