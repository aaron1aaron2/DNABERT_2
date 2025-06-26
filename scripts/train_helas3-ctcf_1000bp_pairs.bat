@echo off

set DATA_PATH=data/helas3_ctcf/1000bp.50ms.text
set MAX_LENGTH=500
set LR=3e-5
set name=DNABERT2_helas3-ctcf_1000bp_pairs

@REM 0.25 * your sequence length

@REM 6 hours 24 min
python finetune/train.py ^
    --model_name_or_path finetune\DNABERT-2-117M ^
    --data_path %DATA_PATH% ^
    --kmer -1 ^
    --run_name %name% ^
    --model_max_length %MAX_LENGTH% ^
    --per_device_train_batch_size 28 ^
    --per_device_eval_batch_size 16 ^
    --gradient_accumulation_steps 1 ^
    --learning_rate %LR% ^
    --num_train_epochs 5 ^
    --fp16 ^
    --logging_steps 100 ^
    --save_steps 2000 ^
    --eval_steps 2000 ^
    --save_model true ^
    --output_dir output/%name% ^
    --cache_dir output/%name%/.cache ^
    --logging_dir output/%name%/log ^
    --save_strategy steps ^
    --evaluation_strategy steps ^
    --eval_accumulation_steps 10 ^
    --warmup_steps 50 ^
    --overwrite_output_dir True ^
    --log_level info ^
    --find_unused_parameters False

python finetune/evaluation.py ^
    --model_name_or_path finetune\DNABERT-2-117M ^
    --data_path  %DATA_PATH% ^
    --kmer -1 ^
    --run_name %name% ^
    --model_max_length %MAX_LENGTH% ^
    --per_device_eval_batch_size 128 ^
    --fp16 ^
    --output_dir output/%name% ^
    --cache_dir output/%name%/.cache ^
    --eval_accumulation_steps 10 ^
    --evaluation_strategy no ^
    --save_strategy no ^
    --log_level info ^
    --find_unused_parameters False