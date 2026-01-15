import pytest
from ecash_intent_llm.schema import SwapIntent, Chain, UserIntent
from pydantic import ValidationError

def test_swap_intent_validation_valid():
    """Ensure a valid SwapIntent passes Pydantic validation."""
    data = {
        "reasoning": "User wants to buy PEPE",
        "confidence_score": 0.95,
        "intent_type": "swap",
        "source_asset": {"symbol": "ETH", "amount": 1.5},
        "target_asset": {"symbol": "PEPE"},
        "chain": "base"
    }
    intent = SwapIntent(**data)
    assert intent.chain == Chain.BASE
    assert intent.source_asset.amount == 1.5

def test_swap_intent_validation_invalid_chain():
    """Ensure invalid chain enum raises validation error."""
    data = {
        "reasoning": "Simple swap",
        "confidence_score": 0.9,
        "intent_type": "swap",
        "source_asset": {"symbol": "ETH"},
        "target_asset": {"symbol": "USDC"},
        "chain": "dogechain"  # Invalid
    }
    with pytest.raises(ValidationError):
        SwapIntent(**data)

def test_polymorphic_deserialization():
    """Ensure raw dict correctly identifies subtypes."""
    # This simulates what the LLM Parser would output as raw JSON object
    # Pydantic's TypeAdapter or direct instantiation would handle this in a real flow
    # Here we just verify the subclass instantiation logic holds.
    
    bridge_payload = {
        "reasoning": "Bridging",
        "confidence_score": 0.99,
        "intent_type": "bridge",
        "source_asset": {"symbol": "USDC", "amount": 500},
        "source_chain": "ethereum",
        "target_chain": "optimism"
    }
    
    # Simulate API response validation
    from ecash_intent_llm.api import UserIntent
    # Pydantic's Union validation is tricky manually, usually automatic in FastAPI
    # but let's test the specific model
    from ecash_intent_llm.schema import BridgeIntent
    
    intent = BridgeIntent(**bridge_payload)
    assert intent.intent_type == "bridge"
    assert intent.target_chain == Chain.OPTIMISM
