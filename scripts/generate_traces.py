import json
import fnmatch
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from datasets import load_dataset

# =========================================================================
# 1. EXPERIMENT CONFIGURATION (from Command Line)
# =========================================================================

@dataclass
class ExperimentConfig:
    """Configuration class to hold all experiment settings."""
    # --- Paths and Files ---
    #   default input file path here
    PROBLEM_FILE_PATH: str = "qwen_total_result/totals.jsonl" 
    OUTPUT_DIR: str = "llama_experiment_outputs"
    
    # --- Model and Tokenizer ---
    # Set your default model here
    # MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # --- Dataset and Filtering ---
    GROUND_TRUTH_DATASET_NAME: str = "WildEval/ZebraLogic"
    GROUND_TRUTH_DATASET_SPLIT: str = "grid_mode"
    PROBLEM_PATTERNS: list[str] = field(default_factory=list)

    # --- Generation Parameters ---
    TEMPERATURE: float = 0.6
    TOP_P: float = 0.95
    MAX_NEW_TOKENS: int = 32000 # 18000
    N_SAMPLES_PER_PROBLEM: int = 10

    # --- Prompting ---
    PROMPT_TEMPLATE: str = (
        "Puzzle:\n{instruction}\n\n"
        "ASSISTANT: "
    )
    
    @property
    def RESULTS_FILENAME(self) -> str:
        # Generate a unique filename based on key parameters
        patterns_str = "_".join(p.replace('*', '') for p in self.PROBLEM_PATTERNS)
        model_name_short = self.MODEL_NAME.split('/')[-1] 
        return f"results_{model_name_short}_{patterns_str}_temp{self.TEMPERATURE}.json"

# =========================================================================
# 2. UTILITY FUNCTIONS (Logging, Parsing, Evaluation, Saving)
# =========================================================================

def setup_logging(config: ExperimentConfig):
    """Configures logging to both console and a file."""
    log_dir = Path(config.OUTPUT_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Capture all levels of logs

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Create console handler with a higher log level (e.g., INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    logging.info(f"Logging initialized. Detailed logs will be saved to: {log_file}")

# (All other utility functions from the previous script remain here, with print -> logging)
def extract_think_blocks(text: str) -> dict:
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)
    if matches: return {'think_content': '\n\n'.join(matches).strip(), 'has_think_block': True}
    return {'think_content': '', 'has_think_block': False}

def extract_json_answer(full_text: str) -> dict | None:
    try:
        tag_pos = full_text.rfind('</think>')
        search_text = full_text[tag_pos:] if tag_pos != -1 else full_text
        json_start_pos = search_text.find('{')
        if json_start_pos == -1: return None
        search_text = search_text[json_start_pos:]
        open_braces = 0
        for i, char in enumerate(search_text):
            if char == '{': open_braces += 1
            elif char == '}': open_braces -= 1
            if open_braces == 0: return json.loads(search_text[:i+1])
        return None
    except (json.JSONDecodeError, IndexError): return None

def evaluate_structured_puzzle(model_output: dict | None, ground_truth: dict) -> bool:
    if not isinstance(model_output, dict) or not isinstance(ground_truth, dict): return False
    try:
        gt_set = {frozenset(str(item).lower() for item in row) for row in ground_truth['rows']}
        model_set = set()
        for h_key, attrs in model_output.items():
            h_num = h_key.lower().split()[-1]
            facts = [h_num] + [str(v).lower() for v in attrs.values()]
            model_set.add(frozenset(facts))
        return model_set == gt_set
    except (KeyError, TypeError, AttributeError, IndexError): return False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_results(data: list, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, cls=NumpyEncoder, indent=2)

def load_results(filepath: Path) -> list:
    if not filepath.exists(): return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, IOError):
        logging.warning(f"Could not load/parse existing results file at {filepath}. Starting fresh.")
        return []

def load_problems_from_jsonl(filepath: str) -> list[dict]:
    problems = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip(): problems.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line {line_num} in {filepath}")
        return problems
    except FileNotFoundError:
        logging.error(f"Input problem file not found at {filepath}")
        return []

class ForceThinkPrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_ids: list, prompt_length: int):
        self.prefix_ids, self.prefix_length, self.prompt_length = prefix_ids, len(prefix_ids), prompt_length
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_len = input_ids.shape[1] - self.prompt_length
        if gen_len < self.prefix_length:
            token = self.prefix_ids[gen_len]
            mask = torch.full_like(scores, -float('inf'))
            mask[:, token] = 0
            scores += mask
        return scores

# =========================================================================
# 3. CORE GENERATION AND EVALUATION FUNCTION
# =========================================================================

def run_sampling_for_problem(problem: dict, config: ExperimentConfig, model, tokenizer, ground_truth_map: dict) -> dict:
    sample_id = problem['id']
    prompt = config.PROMPT_TEMPLATE.format(instruction=problem['instruction'])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    prompt_length = input_ids.shape[1]

    prefix_ids = tokenizer.encode("<think>", add_special_tokens=False)
    force_think_processor = ForceThinkPrefixLogitsProcessor(prefix_ids, prompt_length)

    generation_params = { "max_new_tokens": config.MAX_NEW_TOKENS, "temperature": config.TEMPERATURE, "top_p": config.TOP_P, "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "logits_processor": [force_think_processor] }
    
    all_generations = []
    ground_truth_solution = ground_truth_map.get(sample_id)
    if not ground_truth_solution:
        logging.warning(f"No ground truth found for {sample_id}. Accuracy will be 0.")

    pbar = tqdm(range(config.N_SAMPLES_PER_PROBLEM), desc=f"Sampling for {sample_id}", leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    for run_idx in pbar:
        with torch.no_grad():
            output_ids = model.generate(input_ids, **generation_params)
        
        generated_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
        think_info = extract_think_blocks(generated_text)
        json_answer = extract_json_answer(generated_text)
        is_correct = evaluate_structured_puzzle(json_answer, ground_truth_solution)

        all_generations.append({'run_id': run_idx, 'generated_text': generated_text, 'think_content': think_info['think_content'], 'extracted_json_answer': json_answer, 'accuracy': 1 if is_correct else 0})

    return {'sample_id': sample_id, 'prompt': prompt, 'generations': all_generations}

# =========================================================================
# 4. MAIN EXECUTION SCRIPT
# =========================================================================

def main(config: ExperimentConfig):
    """Main function to run the experiment."""
    setup_logging(config)
    
    logging.info("--- 1. Initializing Model and Tokenizer ---")
    logging.info(f"Loading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    logging.info("\n--- 2. Loading Ground Truth and Input Problems ---")
    gt_ds = load_dataset(config.GROUND_TRUTH_DATASET_NAME, config.GROUND_TRUTH_DATASET_SPLIT)
    ground_truth_map = {item['id']: item['solution'] for item in gt_ds['test']}
    logging.info(f"Loaded {len(ground_truth_map)} ground truth solutions.")

    all_problems_from_file = load_problems_from_jsonl(config.PROBLEM_FILE_PATH)
    if not all_problems_from_file:
        logging.error("No problems loaded from file. Exiting.")
        return
    logging.info(f"Loaded {len(all_problems_from_file)} problems from {config.PROBLEM_FILE_PATH}.")

    problems_to_run = []
    for pattern in config.PROBLEM_PATTERNS:
        filtered = [p for p in all_problems_from_file if 'id' in p and fnmatch.fnmatch(p['id'], pattern)]
        problems_to_run.extend(filtered)
    
    logging.info(f"Found {len(problems_to_run)} problems matching patterns: {config.PROBLEM_PATTERNS}")

    output_filepath = Path(config.OUTPUT_DIR) / config.RESULTS_FILENAME
    all_results = load_results(output_filepath)
    completed_ids = {res['sample_id'] for res in all_results}
    
    if completed_ids:
        logging.info(f"\n--- Resuming experiment: Found {len(completed_ids)} completed problems in {output_filepath} ---")
        problems_to_process = [p for p in problems_to_run if p['id'] not in completed_ids]
    else:
        logging.info(f"\n--- Starting new experiment. Results will be saved to {output_filepath} ---")
        problems_to_process = problems_to_run
        
    if not problems_to_process:
        logging.info("Good! All specified problems have already been processed. Finishing the experiment.")
    else:
        logging.info(f"{len(problems_to_process)} problems remaining to be processed.")

        logging.info("\n--- 3. Starting Generation Loop ---")
        main_pbar = tqdm(problems_to_process, desc="Processing Problems")
        for problem in main_pbar:
            if 'instruction' not in problem or 'id' not in problem:
                logging.warning(f"Skipping malformed problem object: {problem}")
                continue
            
            result = run_sampling_for_problem(problem, config, model, tokenizer, ground_truth_map)
            all_results.append(result)
            
            save_results(all_results, output_filepath)
            main_pbar.set_postfix({"Last saved": problem['id']})

    logging.info("\n--- Experiment Complete ---")
    logging.info(f"All results saved to {output_filepath.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM sampling experiments on logic puzzles.")
    parser.add_argument("--patterns", required=True, nargs='+', help="One or more wildcard patterns to filter problem IDs (e.g., 'lgp-test-2x3*').")
    parser.add_argument("--input-file", required=True, help="Path to the .jsonl file containing the problems.")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Name of the Hugging Face model to use.")
    parser.add_argument("--temp", type=float, default=0.6, help="Sampling temperature for generation.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to generate per problem.")
    parser.add_argument("--output-dir", default="experiment_outputs", help="Directory to save results and logs.")
    args = parser.parse_args()

    config = ExperimentConfig(
        PROBLEM_FILE_PATH=args.input_file,
        PROBLEM_PATTERNS=args.patterns,
        MODEL_NAME=args.model,
        TEMPERATURE=args.temp,
        N_SAMPLES_PER_PROBLEM=args.samples,
        OUTPUT_DIR=args.output_dir
    )
    
    main(config)
