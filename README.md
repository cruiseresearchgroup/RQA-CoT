# Is my model "mind blurting"? 
### Interpreting the dynamics of reasoning tokens with Recurrence Quantification Analysis (RQA)

This repository contains the official implementation for analyzing Large Reasoning Models (LRMs) through the lens of dynamical systems. We propose **Recurrence Quantification Analysis (RQA)** as a non-textual framework to interpret Chain-of-Thought (CoT) processes by analyzing the latent trajectories of hidden states.

Our research demonstrates that temporal RQA signals (like Determinism and Laminarity) provide a high-resolution view of model "stalling" and "overthinking," significantly outperforming token length in predicting task complexity and accuracy.


<img width="611" height="384" alt="Screenshot 2026-01-09 at 11 16 55 am" src="https://github.com/user-attachments/assets/3f6bde8d-a5c7-49e1-b11b-ed5ea91e2f32" />

<img width="630" height="147" alt="Screenshot 2026-01-09 at 11 17 23 am" src="https://github.com/user-attachments/assets/991bf3f7-178f-4c2b-bbd8-e7b1598bb0b8" />


---

## Repository Structure

*   `generate_traces.py`: Phase 1 — LLM inference on ZebraLogic puzzles.
*   `run_rq_analysis.py`: Phase 2 — Hidden state extraction and RQA metric calculation.
*   `evaluate_results.py`: Phase 3 — Statistical evaluation, feature engineering, and classification.
*   `requirements.txt`: List of necessary Python dependencies.

---

## Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/anonymous/RQA-CoT-844C.git
   cd RQA-CoT-844C
   ```

2. **Install dependencies:**
   This project requires specialized libraries for non-linear dynamics analysis (`nolds` and `antropy`).
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

The pipeline is designed to be run in three sequential phases:

### Phase 1: Generate Reasoning Traces
Run the model (e.g., DeepSeek-R1-Distill-Qwen-7B) on ZebraLogic puzzles. This script samples multiple reasoning paths and evaluates the correctness of the final JSON answer.

```bash
python generate_traces.py \
    --input-file data/raw/zebra_logic.jsonl \
    --patterns "2x3*" "3x3*" "4x4*" \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --samples 10 \
    --output-dir experiment_outputs/
```

### Phase 2: Compute Latent RQA Metrics
This script performs a second pass on the generated traces to extract final-layer hidden states ($h_t$). It then computes RQA metrics (DET, LAM, ENTR) globally and over sliding windows.

*Note: Ensure the `INPUT_RESULTS_FILE` inside `run_rq_analysis.py` points to your output from Phase 1.*

```bash
python run_rq_analysis.py
```

### Phase 3: Evaluation & Interpretation
Aggregate the metrics from all complexity levels. This script engineers temporal features (Slopes, DFA, Polynomial curvature) and trains classifiers to predict task complexity and model correctness.

```bash
python evaluate_results.py
```

