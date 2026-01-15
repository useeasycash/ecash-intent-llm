.PHONY: help install dev train serve test lint clean

# Use uv for dependency management
PYTHON := uv run python
PIP := uv pip

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies using uv
	@echo "Installing dependencies..."
	@uv sync

dev: ## Install dev dependencies
	@echo "Installing dev dependencies..."
	@uv sync --dev

train: ## Run the fine-tuning script
	@echo "Starting training process..."
	@$(PYTHON) -m ecash_intent_llm.train --config config/train_config.yaml

serve: ## Start the inference API server
	@echo "Starting inference server..."
	@$(PYTHON) -m uvicorn ecash_intent_llm.api:app --reload --port 8000

test: ## Run tests
	@echo "Running tests..."
	@uv run pytest tests/

lint: ## Run linting and formatting
	@echo "Linting code..."
	@uv run ruff check .
	@uv run black --check .
	@uv run mypy src/

format: ## Format code
	@uv run ruff check --fix .
	@uv run black .

clean: ## Clean sensitive artifacts (not models)
	@echo "Cleaning cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache
