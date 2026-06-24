MODEL_PATH="Qwen/Qwen3-8B"

COMPRESS_PATH="jinandao/NPMCL_Compress"
QUERY_PATH="jinandao/memory_query_lora"
CONVERSATIONS_PATH="jinandao/NPMCL_Conversation"

BASE_DIR="./Configs"
# ===============================

echo "Start Performing..."
python run_configs.py \
    --model_path "$MODEL_PATH" \
    --compress_model_path "$COMPRESS_PATH" \
    --query_model_path "$QUERY_PATH" \
    --conversation_model_path "$CONVERSATIONS_PATH" \
    --base_dir "$BASE_DIR" 