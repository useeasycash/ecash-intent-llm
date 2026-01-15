#!/bin/bash
set -e

# Data Download Script
# Usage: ./scripts/download_data.sh

DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"

echo "📂 Setting up data directories..."
mkdir -p $DATA_DIR
mkdir -p $PROCESSED_DIR

# 1. Download Public Financial Intent Dataset (Mock URL)
echo "⬇️  Downloading open-source financial intents..."
# curl -L https://huggingface.co/datasets/useeasy/financial-intents/resolve/main/train.jsonl -o $DATA_DIR/train.jsonl

# 2. Simulate Local Data Creation if download fails (for Demo)
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "⚠️  Remote dataset not found. Generating synthetic bootstrap data..."
    cat <<EOF > $PROCESSED_DIR/train.jsonl
{"prompt": "Swap 100 USDC for ETH on Base", "completion": "{\\"intent\\": \\"swap\\", \\"source_asset\\": \\"USDC\\", \\"target_asset\\": \\"ETH\\", \\"chain\\": \\"base\\"}"}
{"prompt": "Bridge 500 DAI from Arb to Op", "completion": "{\\"intent\\": \\"bridge\\", \\"source_asset\\": \\"DAI\\", \\"source_chain\\": \\"arbitrum\\", \\"target_chain\\": \\"optimism\\"}"}
EOF
    echo "✅ Synthetic data created at $PROCESSED_DIR/train.jsonl"
fi

echo "✨ Data preparation complete."
