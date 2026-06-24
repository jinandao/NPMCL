MODEL_DIR="Qwen/Qwen3-8B"

# SFT training parameters
SFT_TRAIN_JSON_PATH="./Datasets/compress_en/train/train_data_en.json"
SFT_TEST_JSON_PATH="./Datasets/compress_en/test/test_data_en.json"
SFT_OUTPUT_DIR="./Output/Compress_Memory"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE=2
SFT_GRADIENT_ACCUMULATION_STEPS=2
SFT_NUM_TRAIN_EPOCHS=4
SFT_LEARNING_RATE=2e-5

python memory_compress_sft_en.py \
    --train_json_path "$SFT_TRAIN_JSON_PATH" \
    --test_json_path "$SFT_TEST_JSON_PATH" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --per_device_train_batch_size $SFT_PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $SFT_GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $SFT_NUM_TRAIN_EPOCHS \
    --learning_rate $SFT_LEARNING_RATE \
    --gradient_checkpointing