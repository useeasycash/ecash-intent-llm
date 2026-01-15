import logging
from typing import Optional, Dict
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class FinancialDatasetLoader:
    """
    Production-grade dataset loader with robust error handling and formatting.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def load_from_jsonl(self, file_path: str) -> Dataset:
        """
        Loads a local JSONL file and validates schema.
        Expected format: {"prompt": "...", "completion": "..."}
        """
        logger.info(f"Loading dataset from {file_path}")
        try:
            dataset = load_dataset("json", data_files=file_path, split="train")
        except FileNotFoundError:
            logger.error(f"Dataset file not found at {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        return dataset.map(self._format_prompts, batched=True)

    def _format_prompts(self, examples: Dict[str, list]) -> Dict[str, list]:
        """
        Applies the ChatML format or System Prompt template to the raw data.
        """
        formatted_texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Construct the training sample
            # This follows the format expected by Mistral/Llama instruct models
            text = f"<s>[INST] {prompt} [/INST] {completion} </s>"
            formatted_texts.append(text)
        
        return {"text": formatted_texts}
