MODEL_PATH="Qwen/Qwen3-8B"

COMPRESS_PATH="./Models/Compress/checkpoint-300"
QUERY_PATH="./Models/Query/memory_query_lora"
CONVERSATIONS_PATH="./Models/Conversation/checkpoint-110"

BASE_DIR="./Configs"
# ===============================

echo "Start Performing..."
python run_configs.py \
    --model_path "$MODEL_PATH" \
    --compress_model_path "$COMPRESS_PATH" \
    --query_model_path "$QUERY_PATH" \
    --conversation_model_path "$CONVERSATIONS_PATH" \
    --base_dir "$BASE_DIR" 