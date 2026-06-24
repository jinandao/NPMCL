MODEL_DIR="Qwen/Qwen3-8B"

TRAIN_TRIGGER_JSON_PATH="./Datasets/trigger_en/train/train_data_en.json"
TRAIN_REASONING_JSON_PATH="./Datasets/reasoning_en/train/train_data_en.json"
TRIGGER_REASONING_OUTPUT_DIR="./Output/Trigger_and_Reasoning"
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2
NUM_TRAIN_EPOCHS=5
LEARNING_RATE=2e-5

python memory_trigger_and_reasoning_sft_en.py \
    --train_trigger_json_path "$TRAIN_TRIGGER_JSON_PATH" \
    --train_reasoning_json_path "$TRAIN_REASONING_JSON_PATH" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$TRIGGER_REASONING_OUTPUT_DIR" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gradient_checkpointing


SFT_EXIT_CODE=$?
if [ $SFT_EXIT_CODE -ne 0 ]; then
    echo "SFT train failed: $SFT_EXIT_CODE"
    exit 1
fi

echo "train success！"

TRIGGER_REASONING_MODEL_PATH=""
FINAL_MODEL_DIR="$TRIGGER_REASONING_OUTPUT_DIR/final_model"
if [ -d "$FINAL_MODEL_DIR" ]; then
    TRIGGER_REASONING_MODEL_PATH="$FINAL_MODEL_DIR"
    echo "find final_model path: $TRIGGER_REASONING_MODEL_PATH"
else
    TRIGGER_REASONING_MODEL_PATH="$TRIGGER_REASONING_OUTPUT_DIR"
    echo "not find final_model path, SFT Path: $TRIGGER_REASONING_MODEL_PATH"
fi

TRAIN_CONVERSATIONS_JSON_PATH="./Datasets/conversations_en/train/train_data_en.json"
TEST_CONVERSATIONS_JSON_PATH="./Datasets/conversations_en/test/test_data_en.json"
OUTPUT_DIR="./Output/Conversations"
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=5

echo ""
echo "output dir: $OUTPUT_DIR"

python memory_conversations_en.py \
    --model_dir "$MODEL_DIR" \
    --train_trigger_json_path "$TRAIN_TRIGGER_JSON_PATH" \
    --train_reasoning_json_path "$TRAIN_REASONING_JSON_PATH" \
    --train_conversations_json_path "$TRAIN_CONVERSATIONS_JSON_PATH" \
    --test_conversations_json_path "$TEST_CONVERSATIONS_JSON_PATH" \
    --lora_path "$TRIGGER_REASONING_MODEL_PATH" \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --gradient_checkpointing \

CONVERSATIONS_EXIT_CODE=$?
if [ $CONVERSATIONS_EXIT_CODE -ne 0 ]; then
    echo "train failed: $CONVERSATIONS_EXIT_CODE"
    exit 1
fi