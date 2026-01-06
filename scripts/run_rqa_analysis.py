import json
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from collections import Counter

# 1. ANALYSIS CONFIGURATION
@dataclass
class AnalysisConfig:
    """All settings for the RQA analysis are here."""
    # --- Input and Output Files ---
    INPUT_RESULTS_FILE: str = "experiment_outputs/results_DeepSeek-R1-Distill-Qwen-7B_lgp-test-3x4_temp0.6.json" # 
    OUTPUT_CSV_FILE: str = "Qwen_3x4_sampling_rqa_metrics_output.csv" # change to "Llama_3x4_sampling_rqa_metrics_output.csv"

    # --- Model ---
    MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # change to "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # --- RQA Parameters ---
    SIMILARITY_PERCENTILE: float = 90.0  # Use top 5% most similar states
    MIN_DIAG_LEN: int = 3
    MIN_VERT_LEN: int = 3

    # --- Sliding Window Parameters ---
    WINDOW_SIZE: int = 100  # How many tokens in each window
    STEP_SIZE: int = 15     # How many tokens to slide the window forward

# 2. VALIDATED RQA CORE FUNCTIONS
def calculate_rqa_metrics_validated(recurrence_matrix, min_diag_len=3, min_vert_len=3):
    """Calculates RQA metrics using standard, validated formulas."""
    n = recurrence_matrix.shape[0]
    if n == 0: return {}

    total_recurrent_points = np.sum(recurrence_matrix)
    total_recurrent_points_no_loi = total_recurrent_points - n
    
    if total_recurrent_points_no_loi <= 0:
        return {'recurrence_rate': np.sum(recurrence_matrix) / (n*n), 'determinism': 0.0, 'laminarity': 0.0, 'entropy': 0.0}

    # Diagonal Line Analysis
    diag_line_lengths = []
    for k in range(1, n):
        diag = np.diag(recurrence_matrix, k=k)
        padded = np.concatenate(([0], diag, [0])); diff = np.diff(padded)
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            if length >= min_diag_len: diag_line_lengths.append(length)
    
    points_in_diag_lines = np.sum(diag_line_lengths)
    determinism = (points_in_diag_lines * 2) / total_recurrent_points_no_loi
    
    entropy_diag = 0.0
    if diag_line_lengths:
        counts = np.array(list(Counter(diag_line_lengths).values()))
        probabilities = counts / len(diag_line_lengths)
        entropy_diag = entropy(probabilities, base=2)

    # Vertical Line Analysis
    vert_line_lengths = []
    for j in range(n):
        col = recurrence_matrix[:, j]
        padded = np.concatenate(([0], col, [0])); diff = np.diff(padded)
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            if length >= min_vert_len: vert_line_lengths.append(length)

    points_in_vert_lines = np.sum(vert_line_lengths)
    laminarity = points_in_vert_lines / total_recurrent_points_no_loi
    
    return {'determinism': determinism, 'laminarity': laminarity, 'entropy': entropy_diag}

# 3. ANALYSIS HELPER FUNCTIONS
def get_hidden_states(text: str, model, tokenizer, max_length=16000):
    """Tokenizes text and returns the last hidden states from the model."""
    if not text or not isinstance(text, str):
        return None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    # Return the last hidden layer, squeeze batch dim, and move to CPU as NumPy array
    return outputs.hidden_states[-1].squeeze(0).to(torch.float32).cpu().numpy()

def perform_rqa_on_states(hidden_states, config: AnalysisConfig):
    """Performs a single RQA run on a set of hidden states."""
    if hidden_states is None or hidden_states.shape[0] < config.MIN_DIAG_LEN:
        return {'determinism': np.nan, 'laminarity': np.nan, 'entropy': np.nan}
    
    # Calculate fixed threshold based on percentile
    distances = pdist(hidden_states, metric='cosine')
    threshold = np.percentile(distances, 100 - config.SIMILARITY_PERCENTILE)
    
    # Create recurrence matrix
    distance_matrix = squareform(distances)
    rp = (distance_matrix <= threshold).astype(int)
    np.fill_diagonal(rp, 1)
    
    # Calculate metrics
    return calculate_rqa_metrics_validated(rp, config.MIN_DIAG_LEN, config.MIN_VERT_LEN)

def perform_sliding_window_rqa(hidden_states, config: AnalysisConfig):
    """Performs RQA on a series of sliding windows over the hidden states."""
    if hidden_states is None or hidden_states.shape[0] < config.WINDOW_SIZE:
        return {'det_series': [], 'lam_series': [], 'entr_series': []}

    n_tokens = hidden_states.shape[0]
    det_series, lam_series, entr_series = [], [], []

    for start in range(0, n_tokens - config.WINDOW_SIZE + 1, config.STEP_SIZE):
        end = start + config.WINDOW_SIZE
        window_states = hidden_states[start:end]
        
        try:
            metrics = perform_rqa_on_states(window_states, config)
            det_series.append(float(metrics['determinism']))
            lam_series.append(float(metrics['laminarity']))
            entr_series.append(float(metrics['entropy']))
        except Exception:
            # If any error occurs in a window, append NaN to keep series aligned
            det_series.append(np.nan)
            lam_series.append(np.nan)
            entr_series.append(np.nan)

    return {'det_series': det_series, 'lam_series': lam_series, 'entr_series': entr_series}


if __name__ == "__main__":
    config = AnalysisConfig()
    
    print("--- 1. Loading Model and Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    print(f"\n--- 2. Loading Experiment Results from {config.INPUT_RESULTS_FILE} ---")
    try:
        with open(config.INPUT_RESULTS_FILE, 'r') as f:
            experiment_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found. Please check the path in `AnalysisConfig`.")
        exit()
        
    print(f"Found {len(experiment_data)} problems to analyze.")
    
    all_metrics_results = []
    
    print("\n--- 3. Starting RQA Processing Loop ---")
    # tqdm can be used on any iterable, including a list of dictionaries
    for problem in tqdm(experiment_data, desc="Processing Problems"):
        for generation in problem['generations']:
            
            think_content = generation.get('think_content')
            if len(think_content) == 0:
                # capp at 40000 chars
                think_content = generation.get('generated_text')[:40000]
                
            # This dictionary will store all results for this single generation
            result_row = {
                'id': problem['sample_id'],
                'run_id': generation['run_id'],
                'accuracy': generation['accuracy'],
            }
            
            if think_content and len(think_content.split()) > config.WINDOW_SIZE:
                # This is a potentially slow step
                hidden_states = get_hidden_states(think_content, model, tokenizer)
                
                # Perform Global RQA
                global_metrics = perform_rqa_on_states(hidden_states, config)
                result_row['global_det'] = global_metrics['determinism']
                result_row['global_lam'] = global_metrics['laminarity']
                result_row['global_entr'] = global_metrics['entropy']
                
                # Perform Sliding Window RQA
                window_metrics = perform_sliding_window_rqa(hidden_states, config)
                result_row['window_det_series'] = window_metrics['det_series']
                result_row['window_lam_series'] = window_metrics['lam_series']
                result_row['window_entr_series'] = window_metrics['entr_series']

            else:
                # If content is too short or missing, fill with NaNs
                result_row.update({
                    'global_det': np.nan, 'global_lam': np.nan, 'global_entr': np.nan,
                    'window_det_series': [], 'window_lam_series': [], 'window_entr_series': []
                })
                
            all_metrics_results.append(result_row)

    print("\n--- 4. Saving Results to CSV ---")
    output_path = Path(config.OUTPUT_CSV_FILE)
    df = pd.DataFrame(all_metrics_results)
    df.to_csv(output_path, index=False)

    print(f"Analysis complete. All metrics saved to {output_path.resolve()}")
    print("\n--- First 5 rows of the output data: ---")
    print(df.head().to_string())
