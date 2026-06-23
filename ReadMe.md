
# NPMCL: Non-Parametric Continual Learning

This repository contains the official evaluation scripts and supplementary empirical benchmarks for the **NPMCL (Non-Parametric Meta Continual Learning)** framework.

**Official Paper:** *"NPMCL: A Theoretical Framework for Non-Parametric Continual Learning through Meta-Ability Cultivation"*  
🔗 **DOI Links:** [https://doi.org/10.31224/6634](https://doi.org/10.31224/6634)

---



## ⚙️Environment Setup and Installation

### Create a Conda environment with Python 3.10
conda create -n npmcl python=3.10 -y
### Activate the environment
conda activate npmcl
### Install PyTorch (GPU version)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
### Install core dependencies
pip install -r requirements.txt


## 🚀Training Pipeline and Reproduction
All training workflows are encapsulated in .sh scripts. Upon completion, test set outputs are automatically printed to facilitate direct performance evaluation.

**Hardware Note**: All experiments were conducted and reproduced on a **modified NVIDIA RTX 4090 with 48GB VRAM**. Users may need to adjust the `batch_size`, `max_length`, or file paths to fit their specific experimental environment and hardware constraints.

**1. Memory Compression**, this experiment focuses on condensing extensive dialogue history into high-density logical nodes. 
```plaintext
bash memory_compress_sft_en.sh
```
Expected Result: The model will transform raw dialogues into complete, structure-preserving knowledge chains. Performance is evaluated by a direct comparison between the generated knowledge chains and the manually curated ground truth in the test set, ensuring both informational density and format integrity.

**2. Memory Reasoning**, reproduces the end-to-end inference logic to verify the model’s ability to execute rigorous reasoning using external knowledge.
```plaintext
bash memory_reasoning_sft_en.sh
```
Expected Result: The inference results will be printed in the log files, which can be manually cross-referenced and compared against the ground truth in the test set.

**3. Memory-based Synthesis Conversation**，Verifies the model's integrity in utilizing retrieved memories for multi-round dialogue. It employs a Two-Phase Curriculum Learning strategy:
Phase 1 (Atomic Internalization): Single-turn SFT to internalize core "meta-abilities" of memory triggering and constrained reasoning.
Phase 2 (Collaborative Orchestration): Multi-round dialogue SFT with a subset of Phase 1 data retained. This ensures the model achieves flexible triggering and accurate reasoning within structured, multi-turn conversation templates.
```plaintext
   bash memory_trigger_reasoning_conversations.sh
```
Expected Result: AI-generated responses based on retrieved context. The test set includes human-curated Chain-of-Thought (think) traces and ground-truth responses to evaluate synthesis performance and reasoning accuracy.

## 🧪Demo: Non-parametric Learning
A rapid demonstration script is provided to observe LLM behavior when encountering novel knowledge (e.g., counterfactual formulas or Azeroth-specific rules).
```plaintext
bash run_configs.sh
```
#### Comparison: The execution log sequentially prints the Reference (Ground-Truth) output followed by the Model's Generated Response, allowing for a direct assessment of accuracy and consistency.


## 🧪 Controlled Baseline Experiments
﻿
The `ControlledExperiment/` directory contains the standardized evaluation scripts for benchmarking against competitive memory frameworks (**MemGPT** and **Mem0**).
﻿
```text
.
└── ControlledExperiment/
├── mem0_control_eval.py       # Standardized Mem0 replication pipeline
└── memgpt_control_eval.py     # Local MemGPT client step execution loop
```

### 📋 Prerequisites & Environment Setup
﻿
To fully replicate the baseline benchmarks, ensure your local execution environment meets the following specific requirements:
﻿
* **Local LLM Engine:** A localized **Ollama** environment must be properly installed and configured on your Windows host. Ensure both `qwen3:8b-q8_0` (or your designated size variant) and `qwen3-embedding:0.6b` are pulled and active via local endpoints (`http://localhost:11434`).
* **Framework Version Locks:** The baselines rely on strict legacy and development releases for local multi-turn compatibility. Install the exact verified versions via `pip`:
```bash
pip install pymemgpt-nightly==0.4.0.dev20240919104109
pip install mem0ai==2.0.2