# ecash-intent-llm 🧠

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Powered by](https://img.shields.io/badge/Powered%20by-EasyCash-success)](https://useeasy.cash)

**Production-grade LLM inference engine and fine-tuning pipeline for the EasyCash Protocol.**

This repository contains the "brain" of the EasyCash Agent. It is responsible for parsing unstructured natural language user intents (e.g., *"Bridge 500 USDC to Base and swap for DEGEN"*) into structured, executable JSON instructions that the `easy-agent-router` can execute.

## 🚀 Features

*   **Intent Recognition**: Fine-tuned logic to distinguish between Swap, Bridge, Transfer, and obscure financial intents.
*   **Structured Output**: Guarantees JSON output compliant with the EasyCash Protocol Schema.
*   **Efficient Training**: Uses LoRA (Low-Rank Adaptation) via PEFT to fine-tune large foundation models (Llama 3, Mistral) on consumer hardware.
*   **Production API**: FastAPI-based inference server ready for deployment.
*   **UV Managed**: Blazing fast dependency management.

## 🛠 Installation

We use [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/useeasycash/ecash-intent-llm.git
    cd ecash-intent-llm
    ```

2.  **Install dependencies**
    ```bash
    make install
    ```

## 🏃‍♂️ Usage

### 1. Training (Fine-tuning)

Prepare your dataset in `data/raw/` and configure parameters in `config/train_config.yaml`.

```bash
make train
```

This will output adapter weights to `models/adapters/`.

### 2. Inference API

Start the REST API server to serve predictions:

```bash
make serve
```

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/intent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Get me 500 dollars worth of ETH on Arbitrum using my USDC"}'
```

**Example Response:**

```json
{
  "intent": "bridge_swap",
  "source_asset": "USDC",
  "target_asset": "ETH",
  "target_chain": "Arbitrum",
  "amount_in": 500,
  "confidence": 0.98
}
```

## 📂 Project Structure

```
ecash-intent-llm/
├── config/                 # Configuration files (YAML)
├── data/                   # Dataset management
│   ├── raw/                # Raw JSONL datasets
│   └── processed/          # Tokenized datasets
├── models/                 # Model artifacts
├── src/
│   └── ecash_intent_llm/   # Source code
│       ├── api.py          # FastAPI application
│       ├── model.py        # Model wrapper & logic
│       ├── train.py        # Training pipeline
│       └── utils.py        # Helper functions
├── tests/                  # Unit tests
├── Makefile                # Command runner
├── pyproject.toml          # Dependencies (uv)
└── README.md               # Documentation
```

## 🤝 Contributing

We welcome contributions to the dataset! Improving the model's understanding of crypto-slang improves the entire protocol.

1.  Fork the repo
2.  Add examples to `data/raw/community_intents.jsonl`
3.  Submit a PR

## 🛡 License

MIT Public License.
