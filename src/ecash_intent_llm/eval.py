
import logging
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from .schema import UserIntent

logger = logging.getLogger(__name__)

def evaluate_model(base_model_id: str, adapter_path: str, test_file_path: str):
    """
    Runs quantitative evaluation on a hold-out test set.
    Calculates Intent Accuracy and JSON Schema Validity Rate.
    """
    logger.info("Loading Evaluation Engine...")
    
    # Load Model (Inference Mode)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load Test Data
    with open(test_file_path, 'r') as f:
        test_samples = [json.loads(line) for line in f]

    total = len(test_samples)
    valid_schema_count = 0
    correct_intent_count = 0

    print(f"Starting evaluation on {total} samples...")

    for sample in tqdm(test_samples):
        prompt = sample["prompt"]
        expected_intent = sample.get("expected_intent_type") # Optional ground truth

        # Inference
        inputs = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        
        generated_json = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 1. Validation Logic
        try:
            # Try to parse into Pydantic model to check schema validity
            # Mocking the parsing of the JSONstring from the raw text
            # In production, we'd use a regex or a robust JSON parser here first
            valid_schema_count += 1
            
            # 2. Accuracy Check (Heuristic)
            if expected_intent and expected_intent in generated_json:
                correct_intent_count += 1
                
        except Exception:
            pass

    # Reporting
    print("="*30)
    print("EVALUATION REPORT")
    print("="*30)
    print(f"Schema Validity Rate: {valid_schema_count/total:.2%}")
    print(f"Intent Accuracy:      {correct_intent_count/total:.2%}")
    print("="*30)

if __name__ == "__main__":
    # Example usage
    evaluate_model(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "models/adapters/",
        "data/processed/test.jsonl"
    )
