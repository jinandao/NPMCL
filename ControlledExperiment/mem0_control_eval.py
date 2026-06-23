import os
os.environ["DISABLE_TELEMETRY"] = "true"
os.environ["MEM0_TELEMETRY"] = "false"
from mem0 import Memory
import ollama
import json

# ===================== 你的原有配置（完全不用改）=====================
os.environ.pop("OPENAI_API_KEY", None)
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen3:8b-q8_0",           # 推理模型，确保已 ollama pull llama3
            # "base_url": "http://localhost:11434",
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "qwen3-embedding:0.6b", # 嵌入模型，确保已下载
            # "endpoint": "http://localhost:11434",
            "embedding_dims": 1024,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": "mem0_local_db",
            "embedding_model_dims": 1024  # 2.0版本正确的维度参数名，直接写在这里
        }
    },
}

m = Memory.from_config(config)
USER_ID = "test_import_user"
SIMILARITY_THRESHOLD = 0.5


# ===================== 等价于原生chat的封装 =====================
def chat_with_memory(query: str, auto_save: bool = True):
    # 1. 自动检索记忆
    search_result = m.search(query, filters={'user_id': USER_ID})
    filtered_memories = [
        r["memory"] for r in search_result["results"]
        if r["score"] >= SIMILARITY_THRESHOLD
    ]

    # 2. 拼Prompt生成回答
    memory_context = "\n".join([f"- {mem}" for mem in filtered_memories]) if filtered_memories else "No relevant memories found."
    prompt = f"""
You must strictly answer the user's question based on the background memories below. Do not make up information that is not in the memories.
If there is no relevant content in the memories, directly say "No relevant memory found".

【Background Memories】
{memory_context}

【User Question】
{query}

Please give a concise and accurate answer.
    """

    # 3. 调用本地LLM
    response = ollama.chat(model="qwen3:8b-q8_0", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    # 4. 自动保存本轮对话到记忆
    if auto_save:
        m.add(
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ],
            user_id=USER_ID
        )

    return answer, filtered_memories


# ===================== 批量导入历史对话 =====================
print("正在批量导入历史对话...")

FILE_TEACH = "./demos_knowledge_teach_en.json"
with open(FILE_TEACH, 'r', encoding='utf-8') as f:
    data_teach = json.load(f)
convs = []
for i in range(0, len(data_teach)):
    print("开始压缩第", i + 1 ,"条数据")
    conversations = data_teach[i]["conversations"]
    m.add(messages=conversations, user_id=USER_ID)
print("导入完成！\n" + "=" * 60)

FILE_USE_KNOWLEDGE = "./demos_knowledge_use_conversations_en.json"
questions=[]
with open(FILE_USE_KNOWLEDGE, 'r', encoding='utf-8') as f:
    data_use = json.load(f)
    for i in range(0, len(data_use)):
        questions.append(data_use[i]['conversations'][2]['content'])

# ===================== 测试对话 =====================

print("【记忆对话测试】")
for i in range(0, len(questions)):
    query = questions[i]
    print(f"\n--- 测试 {i} ---")
    print(f"用户问题：{query}")

    answer, used_memories = chat_with_memory(query, auto_save=False)

    print(f"匹配到的记忆：{used_memories}")
    print(f"AI 回答：{answer}")
